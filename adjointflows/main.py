from tools import ConfigManager
from tools import GLOBAL_PARAMS
from tools import RunStateManager
from workflow import WorkflowController
from workflow.execution_control import finalize_after_forward_loop
from workflow.execution_control import fail_preflight
from workflow.execution_control import mark_failed_state_file
from workflow.execution_control import run_forward_attempt
from workflow.execution_control import run_inversion_stage
from workflow.execution_control import run_postprocess_stage
from workflow.execution_control import update_state
from workflow.execution_control import validate_and_normalize_workflow_controls
import sys
import logging
import datetime


def _set_config_value(config_manager, key, value):
    """Set a dotted config key in-memory so downstream objects read updated values."""
    keys = key.split(".")
    cursor = config_manager.config
    for part in keys[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[keys[-1]] = value


def setup_logging():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    debug_logger = logging.getLogger("debug_logger")
    debug_logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)

    debug_file_handler = logging.FileHandler(f"logger/debug_{timestamp}.log", mode="w")
    debug_file_handler.setLevel(logging.DEBUG)
    debug_file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    debug_file_handler.setFormatter(debug_file_formatter)

    debug_logger.addHandler(console_handler)
    debug_logger.addHandler(debug_file_handler)

    result_logger = logging.getLogger("result_logger")
    result_logger.setLevel(logging.INFO)

    result_file_handler = logging.FileHandler(f"logger/result_{timestamp}.log", mode="w")
    result_file_handler.setLevel(logging.INFO)
    result_file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    result_file_handler.setFormatter(result_file_formatter)

    result_logger.addHandler(result_file_handler)
    result_logger.addHandler(console_handler)
    result_logger.addHandler(debug_file_handler)

    debug_logger.propagate = False
    result_logger.propagate = False


def main():
    
    # ---------------------------------------------------------------------------
    # 0) Load config, setup logging, and initialize run state manager
    # ---------------------------------------------------------------------------
    setup_logging()
    debug_logger = logging.getLogger("debug_logger")
    result_logger = logging.getLogger("result_logger")

    debug_logger.info("Start the adjoint tomography workflow...")
    config = ConfigManager("config.yaml")
    config.load()

    max_attempts = int(config.get("inversion.max_fail"))
    min_improvement_pct = float(config.get("inversion.min_improvement_pct", 0.0))
    max_effective_updates = int(config.get("inversion.max_effective_updates", 0))
    raw_sd_runs_num = config.get("inversion.sd_runs_num")
    run_mode = config.get("setup.workflow.run_mode")
    inversion_execution_mode = str(
        config.get("setup.workflow.inversion_execution_mode", "single")
    ).strip().lower()
    start_from_stage = config.get("setup.workflow.start_from_stage")
    end_at_stage = config.get("setup.workflow.end_at_stage")
    forward_stop_at_user = config.get("setup.workflow.forward_stop_at")
    current_model_num = int(config.get("setup.model.current_model_num"))
    do_mesh = bool(config.get("setup.model.do_mesh"))
    do_generate = bool(config.get("setup.model.do_generate"))

    run_state_manager = RunStateManager(base_dir=GLOBAL_PARAMS["base_dir"])
    try:
        sd_runs_num = int(raw_sd_runs_num)
    except (TypeError, ValueError):
        fail_preflight(
            result_logger,
            run_state_manager,
            "inversion.sd_runs_num must be a positive integer (> 0).",
        )

    if sd_runs_num <= 0:
        fail_preflight(
            result_logger,
            run_state_manager,
            "inversion.sd_runs_num must be a positive integer (> 0).",
        )

    if max_effective_updates < 0:
        fail_preflight(
            result_logger,
            run_state_manager,
            "inversion.max_effective_updates must be >= 0.",
        )

    update_state(
        run_state_manager,
        status="RUNNING",
        stage="INIT",
        current_model_num=current_model_num,
        extra={
            "run_mode": run_mode,
            "inversion_execution_mode": inversion_execution_mode,
            "start_from_stage": start_from_stage,
            "end_at_stage": end_at_stage,
            "forward_stop_at": forward_stop_at_user,
            "max_attempts": max_attempts,
            "min_improvement_pct": min_improvement_pct,
            "max_effective_updates": max_effective_updates,
            "sd_runs_num": sd_runs_num,
        },
    )

    stage_order, start_index, end_index, forward_stop_at = validate_and_normalize_workflow_controls(
        inversion_execution_mode=inversion_execution_mode,
        run_mode=run_mode,
        start_from_stage=start_from_stage,
        end_at_stage=end_at_stage,
        forward_stop_at_user=forward_stop_at_user,
        result_logger=result_logger,
        run_state_manager=run_state_manager,
    )
    do_measurement = forward_stop_at != "synthetics"
    do_adjoint = forward_stop_at not in ("misfit", "synthetics")

    model_cycle = 0
    effective_updates = 0
    result_logger.info(f"Workflow: User choose to start from {start_from_stage}")

    while True:
        model_cycle += 1
        _set_config_value(config, "setup.model.current_model_num", int(current_model_num))
        workflow_controller = WorkflowController(config=config, global_params=GLOBAL_PARAMS)
        workflow_controller.setup_for_fail()

        if model_cycle == 1 and start_index > stage_order["forward"]:
            workflow_controller.move_to_other_directory(folder_to_move="specfem")
            workflow_controller.load_specfem_params_without_generation()

        attempt = 0
        misfit_reduced = False
        stop_due_to_small_improvement = False
        misfit_result = None
        cycle_do_generate = do_generate if model_cycle == 1 else True

        while (
            not misfit_reduced
            and attempt < max_attempts
            and start_index <= stage_order["forward"]
            and stage_order["forward"] <= end_index
        ):
            attempt += 1
            attempt_result = run_forward_attempt(
                workflow_controller,
                attempt=attempt,
                current_model_num=current_model_num,
                do_generate=cycle_do_generate,
                do_mesh=do_mesh,
                run_mode=run_mode,
                do_adjoint=do_adjoint,
                do_measurement=do_measurement,
                forward_stop_at=forward_stop_at,
                min_improvement_pct=min_improvement_pct,
                run_state_manager=run_state_manager,
                result_logger=result_logger,
            )
            do_mesh = attempt_result.get("do_mesh_next", False)

            if "return_code" in attempt_result:
                return attempt_result["return_code"]
            if attempt_result.get("break_loop"):
                break

            misfit_result = attempt_result.get("misfit_result")
            if attempt_result.get("misfit_reduced"):
                misfit_reduced = True
                stop_due_to_small_improvement = attempt_result.get("stop_due_to_small_improvement", False)

        maybe_return_code = finalize_after_forward_loop(
            misfit_reduced=misfit_reduced,
            attempt=attempt,
            max_attempts=max_attempts,
            stop_due_to_small_improvement=stop_due_to_small_improvement,
            min_improvement_pct=min_improvement_pct,
            misfit_result=misfit_result,
            run_state_manager=run_state_manager,
            result_logger=result_logger,
        )
        if maybe_return_code is not None:
            return maybe_return_code

        maybe_return_code = run_postprocess_stage(
            workflow_controller,
            start_index=start_index,
            end_index=end_index,
            stage_order=stage_order,
            attempt=attempt,
            forward_stop_at=forward_stop_at,
            run_state_manager=run_state_manager,
            result_logger=result_logger,
        )
        if maybe_return_code is not None:
            return maybe_return_code

        run_inversion_stage(
            workflow_controller,
            start_index=start_index,
            end_index=end_index,
            stage_order=stage_order,
            attempt=attempt,
            run_state_manager=run_state_manager,
        )

        if inversion_execution_mode != "continuous":
            update_state(
                run_state_manager,
                status="COMPLETED",
                stage="DONE",
                attempt=attempt,
                current_model_num=current_model_num,
                decision="FINISH_WORKFLOW",
            )
            return 0

        effective_updates += 1
        next_model_num = current_model_num + 1
        if max_effective_updates > 0 and effective_updates >= max_effective_updates:
            update_state(
                run_state_manager,
                status="STOPPED",
                stage="LOOP_CONTROL",
                attempt=attempt,
                current_model_num=next_model_num,
                decision="STOP_MAX_EFFECTIVE_UPDATES",
                extra={
                    "effective_updates": effective_updates,
                    "max_effective_updates": max_effective_updates,
                    "model_cycle": model_cycle,
                },
            )
            result_logger.info(
                "STOP: reached max_effective_updates="
                f"{max_effective_updates}. Latest model is m{next_model_num:03d}."
            )
            return 0

        current_model_num = next_model_num
        do_generate = True
        do_mesh = False
        update_state(
            run_state_manager,
            status="RUNNING",
            stage="LOOP_CONTROL",
            attempt=attempt,
            current_model_num=current_model_num,
            decision="CONTINUE_NEXT_MODEL",
            extra={
                "model_cycle": model_cycle,
                "effective_updates": effective_updates,
                "max_effective_updates": max_effective_updates,
            },
        )
        result_logger.info(
            f"CONTINUOUS mode: proceed to next model m{current_model_num:03d} (cycle {model_cycle + 1})."
        )


if __name__ == "__main__":
    try:
        main_status = main()
    except Exception as exc:
        mark_failed_state_file(base_dir=GLOBAL_PARAMS["base_dir"], exc=exc)
        logging.getLogger("debug_logger").exception("Workflow failed with an unhandled exception.")
        raise
    sys.exit(main_status)
