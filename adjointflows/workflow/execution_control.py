import datetime
import json
import os

"""Execution control helpers for workflow stage orchestration and run-state updates."""


def update_state(
    run_state_manager,
    *,
    status,
    stage,
    attempt=None,
    current_model_num=None,
    decision=None,
    extra=None,
):
    """Write a normalized state update payload into run_state.json."""
    payload = {
        "status": status,
        "stage": stage,
    }
    if attempt is not None:
        payload["attempt"] = attempt
    if current_model_num is not None:
        payload["current_model_num"] = current_model_num
    if decision is not None:
        payload["decision"] = decision
    if extra:
        payload.update(extra)
    run_state_manager.update(**payload)


def fail_preflight(result_logger, run_state_manager, error_message):
    """Record a preflight validation failure to state, then raise ValueError."""
    result_logger.error(error_message)
    update_state(
        run_state_manager,
        status="FAILED",
        stage="INIT",
        extra={
            "error_type": "ValueError",
            "error_message": error_message,
        },
    )
    raise ValueError(error_message)


def validate_and_normalize_workflow_controls(
    inversion_execution_mode,
    run_mode,
    start_from_stage,
    end_at_stage,
    forward_stop_at_user,
    result_logger,
    run_state_manager,
):
    """Validate workflow control fields and return normalized stage controls.

    Returns:
        tuple: (stage_order, start_index, end_index, forward_stop_at)
    """
    stage_order = {
        "forward": 1,
        "postprocess": 2,
        "inversion": 3,
    }

    if inversion_execution_mode not in ("single", "continuous"):
        fail_preflight(
            result_logger,
            run_state_manager,
            "setup.workflow.inversion_execution_mode must be 'single' or 'continuous'.",
        )

    if start_from_stage not in stage_order or end_at_stage not in stage_order:
        fail_preflight(
            result_logger,
            run_state_manager,
            "start_from_stage/end_at_stage must be one of: forward, postprocess, inversion.",
        )

    if inversion_execution_mode == "continuous":
        if run_mode != "pipeline":
            fail_preflight(
                result_logger,
                run_state_manager,
                "For continuous mode, setup.workflow.run_mode must be 'pipeline'.",
            )
        if start_from_stage != "forward" or end_at_stage != "inversion":
            fail_preflight(
                result_logger,
                run_state_manager,
                "For continuous mode, start_from_stage/end_at_stage must be "
                "'forward'/'inversion'.",
            )
        if forward_stop_at_user != "full":
            fail_preflight(
                result_logger,
                run_state_manager,
                "For continuous mode, setup.workflow.forward_stop_at must be 'full'.",
            )

    start_index = stage_order[start_from_stage]
    end_index = stage_order[end_at_stage]
    if start_index > end_index:
        fail_preflight(
            result_logger,
            run_state_manager,
            f"start_from_stage ({start_from_stage}) cannot be after end_at_stage ({end_at_stage}).",
        )

    forward_stop_at = forward_stop_at_user
    if inversion_execution_mode == "single" and end_at_stage != "forward":
        forward_stop_at = "full"

    return stage_order, start_index, end_index, forward_stop_at


def mark_failed_state_file(base_dir, exc):
    """Best-effort fallback: mark run_state.json as FAILED on unhandled exception."""
    state_path = os.path.join(base_dir, "TOMO", ".state", "run_state.json")
    try:
        if os.path.isfile(state_path):
            with open(state_path, "r") as f:
                state = json.load(f)
        else:
            state = {"schema_version": "1.0"}
        state.update(
            {
                "status": "FAILED",
                "stage": state.get("stage", "UNKNOWN"),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "updated_at": datetime.datetime.now(datetime.timezone.utc)
                .astimezone()
                .isoformat(timespec="seconds"),
            }
        )
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2, sort_keys=True)
    except Exception:
        pass


def run_forward_attempt(
    workflow_controller,
    *,
    attempt,
    current_model_num,
    do_generate,
    do_mesh,
    run_mode,
    do_adjoint,
    do_measurement,
    forward_stop_at,
    min_improvement_pct,
    run_state_manager,
    result_logger,
):
    """Run one forward attempt and return a decision payload for the main loop.

    Returns a dict that may include:
    - return_code: immediate process exit code
    - break_loop: stop forward loop but continue later stages
    - misfit_reduced / stop_due_to_small_improvement / misfit_result
    - do_mesh_next: mesh flag for next attempt
    """
    update_state(
        run_state_manager,
        status="RUNNING",
        stage="FORWARD",
        attempt=attempt,
        current_model_num=current_model_num,
        decision="RUN_FORWARD",
    )

    workflow_controller.move_to_other_directory(folder_to_move="specfem")
    if do_generate or attempt > 1:
        workflow_controller.generate_model(mesh_flag=do_mesh)
    else:
        workflow_controller.load_specfem_params_without_generation()

    if run_mode == "flexwin_test":
        workflow_controller.run_flexwin_test_datasets()
        update_state(
            run_state_manager,
            status="STOPPED",
            stage="FORWARD",
            decision="STOP_FLEXWIN_TEST",
        )
        return {"return_code": 0, "do_mesh_next": False}

    workflow_controller.run_all_datasets(do_adjoint=do_adjoint, do_measurement=do_measurement)

    if forward_stop_at == "gradient":
        result_logger.info("forward_stop_at='gradient': forward terminated after gradient computation.")
        update_state(
            run_state_manager,
            status="STOPPED",
            stage="FORWARD",
            attempt=attempt,
            decision="STOP_FORWARD_AT_GRADIENT",
        )
        return {"break_loop": True, "do_mesh_next": False}

    if forward_stop_at == "misfit":
        result_logger.info("Only compute misfit as requested by user. STOP!.")
        update_state(
            run_state_manager,
            status="STOPPED",
            stage="FORWARD",
            attempt=attempt,
            decision="STOP_FORWARD_AT_MISFIT",
        )
        return {"return_code": 0, "do_mesh_next": False}

    if forward_stop_at == "synthetics":
        result_logger.info("Only create synthetic waveforms as requested by user. STOP!.")
        update_state(
            run_state_manager,
            status="STOPPED",
            stage="FORWARD",
            attempt=attempt,
            decision="STOP_FORWARD_AT_SYNTHETICS",
        )
        return {"return_code": 0, "do_mesh_next": False}

    misfit_result = workflow_controller.misfit_check(min_improvement_pct=min_improvement_pct)
    update_state(
        run_state_manager,
        status="RUNNING",
        stage="MISFIT_CHECK",
        attempt=attempt,
        current_model_num=current_model_num,
        decision=misfit_result.get("decision"),
        extra={
            "misfit_current": misfit_result.get("current_misfit"),
            "misfit_previous": misfit_result.get("previous_misfit"),
            "misfit_improvement_pct": misfit_result.get("improvement_pct"),
            "stop_due_to_small_improvement": misfit_result.get("stop_due_to_small_improvement", False),
        },
    )
    run_state_manager.record_iteration(
        iter_index=current_model_num,
        attempt=attempt,
        payload={
            "status": "PASS" if misfit_result.get("misfit_reduced") else "RETRY",
            "decision": misfit_result.get("decision"),
            "metrics": {
                "misfit_current": misfit_result.get("current_misfit"),
                "misfit_previous": misfit_result.get("previous_misfit"),
                "misfit_improvement_pct": misfit_result.get("improvement_pct"),
                "threshold_pct": misfit_result.get("threshold_pct"),
            },
            "stop_due_to_small_improvement": misfit_result.get("stop_due_to_small_improvement", False),
        },
    )

    if misfit_result.get("misfit_reduced"):
        return {
            "misfit_reduced": True,
            "stop_due_to_small_improvement": misfit_result.get("stop_due_to_small_improvement", False),
            "misfit_result": misfit_result,
            "do_mesh_next": False,
        }

    workflow_controller.reupdate_model_if_misfit_not_reduced()
    update_state(
        run_state_manager,
        status="RETRY",
        stage="MISFIT_CHECK",
        attempt=attempt,
        decision="ROLLBACK_AND_RETRY",
    )
    return {
        "misfit_reduced": False,
        "stop_due_to_small_improvement": False,
        "misfit_result": misfit_result,
        "do_mesh_next": False,
    }


def finalize_after_forward_loop(
    *,
    misfit_reduced,
    attempt,
    max_attempts,
    stop_due_to_small_improvement,
    min_improvement_pct,
    misfit_result,
    run_state_manager,
    result_logger,
):
    """Handle forward-loop termination conditions.

    Returns:
        int | None: 0 means stop workflow now; None means continue.
    """
    if not misfit_reduced and attempt == max_attempts:
        result_logger.warning("STOP: Reached max attempts without reducing misfit.")
        update_state(
            run_state_manager,
            status="STOPPED",
            stage="MISFIT_CHECK",
            attempt=attempt,
            decision="STOP_MAX_ATTEMPTS",
        )
        return 0

    if stop_due_to_small_improvement:
        result_logger.info(
            f"STOP: misfit improvement is below threshold ({min_improvement_pct:.3f}%). "
            "Skip postprocess/inversion."
        )
        update_state(
            run_state_manager,
            status="STOPPED",
            stage="MISFIT_CHECK",
            attempt=attempt,
            decision="STOP_SMALL_IMPROVEMENT",
            extra={
                "threshold_pct": min_improvement_pct,
                "misfit_improvement_pct": misfit_result.get("improvement_pct") if misfit_result else None,
            },
        )
        return 0

    return None


def run_postprocess_stage(
    workflow_controller,
    *,
    start_index,
    end_index,
    stage_order,
    attempt,
    forward_stop_at,
    run_state_manager,
    result_logger,
):
    """Run postprocess stage if selected by stage window.

    Returns:
        int | None: 0 when user requested stop after gradient; otherwise None.
    """
    if not (start_index <= stage_order["postprocess"] <= end_index):
        return None

    update_state(
        run_state_manager,
        status="RUNNING",
        stage="POSTPROCESS",
        attempt=attempt,
        decision="RUN_POSTPROCESS",
    )
    workflow_controller.move_to_other_directory(folder_to_move="specfem")
    workflow_controller.create_misfit_kernel_each_dataset()

    if forward_stop_at == "gradient":
        result_logger.info("Only compute gradient as requested by user. Stop after post-processing step.")
        update_state(
            run_state_manager,
            status="STOPPED",
            stage="POSTPROCESS",
            attempt=attempt,
            decision="STOP_AFTER_GRADIENT_POSTPROCESS",
        )
        return 0

    return None


def run_inversion_stage(
    workflow_controller,
    *,
    start_index,
    end_index,
    stage_order,
    attempt,
    run_state_manager,
):
    """Run inversion stage if selected by stage window."""
    if not (start_index <= stage_order["inversion"] <= end_index):
        return

    update_state(
        run_state_manager,
        status="RUNNING",
        stage="INVERSION",
        attempt=attempt,
        decision="RUN_INVERSION",
    )
    workflow_controller.move_to_other_directory(folder_to_move="adjointflows")
    workflow_controller.do_iteration()
    workflow_controller.cleanup_after_inversion()
