#%%
import os
import sys

import numpy as np
import pandas as pd
import pygmt

# Allow running from utils/ without PYTHONPATH preconfigured.
try:
    from tools.dataset_loader import (
        deep_merge,
        get_by_path,
        load_dataset_config,
        resolve_dataset_list_path,
    )
except ModuleNotFoundError:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    adjointflows_dir = os.path.abspath(os.path.join(script_dir, os.pardir, "adjointflows"))
    if adjointflows_dir not in sys.path:
        sys.path.insert(0, adjointflows_dir)
    from tools.dataset_loader import (
        deep_merge,
        get_by_path,
        load_dataset_config,
        resolve_dataset_list_path,
    )


def _build_dataset_entries(dataset_config):
    defaults = dataset_config.get("defaults", {})
    datasets = dataset_config.get("datasets", [])
    merged = []
    for entry in datasets:
        if not isinstance(entry, dict):
            continue
        merged.append(deep_merge(defaults, entry))
    return merged


def _measure_dir_name(dataset_name):
    if dataset_name:
        return f"MEASURE_{dataset_name}"
    return "MEASURE"


def get_chi(evlst, model_num, dataset_name=None):
    global tomo_dir

    measure_dir = f"{tomo_dir}/m{model_num:03d}/{_measure_dir_name(dataset_name)}/adjoints"
    evt_df = pd.read_csv(evlst, header=None, sep=r"\s+")
    chi_df = pd.DataFrame()
    missing_chi = []
    for evt in evt_df[0]:
        adjoints_dir = f"{measure_dir}/{evt}"
        misfit_file = f"{adjoints_dir}/window_chi"
        if not os.path.isfile(misfit_file):
            missing_chi.append(evt)
            continue
        try:
            tmp_df = pd.read_csv(misfit_file, sep=r"\s+", header=None)
        except Exception as exc:
            print(f"Skip {evt}: failed to read {misfit_file} ({exc})")
            missing_chi.append(evt)
            continue
        chi_df = pd.concat([chi_df, tmp_df])

    if missing_chi:
        missing_str = ", ".join(missing_chi)
        print(f"Missing window_chi for events: {missing_str}. Please check window selection.")
    if chi_df.empty:
        print("No window_chi files found; misfit set to 0.")
        return 0.0

    chi_filtered_df = chi_df[(chi_df[28] != 0.0) | (chi_df[29] != 0.0)]
    if chi_filtered_df.empty:
        print("No valid windows; misfit set to 0.")
        return 0.0
    total_misfit = chi_filtered_df[28].sum()
    win_num = len(chi_filtered_df)
    if win_num == 0:
        print("No valid windows; misfit set to 0.")
        return 0.0
    average_misfit = round(total_misfit / win_num, 5)
    print(f"chi_arr: {average_misfit}")

    return average_misfit


def get_misfit_list(model_beg, model_end, dataset_entries, base_dir, default_evlst_name=None):
    misfit_list = []
    for model_num in range(model_beg, model_end + 1):
        total_misfit = 0.0
        total_weight = 0.0
        for dataset_entry in dataset_entries:
            dataset_name = get_by_path(dataset_entry, "name", default="dataset")
            weight = float(get_by_path(dataset_entry, "inversion.weight", 1.0))
            evlst = resolve_dataset_list_path(
                base_dir,
                dataset_entry,
                "list.evlst",
                "evlst",
                default=default_evlst_name,
                required=True,
            )
            misfit = get_chi(evlst, model_num, dataset_name=dataset_name)
            total_misfit += misfit * weight
            total_weight += weight

        if total_weight == 0.0:
            misfit_list.append(0.0)
        else:
            misfit_list.append(total_misfit / total_weight)
    return misfit_list


def get_improvement_pct_list(misfit_list):
    improvement_pct_list = [None]
    for idx in range(1, len(misfit_list)):
        previous_misfit = misfit_list[idx - 1]
        current_misfit = misfit_list[idx]
        if previous_misfit == 0.0:
            improvement_pct_list.append(None if current_misfit != 0.0 else 0.0)
            continue
        improvement_pct = ((previous_misfit - current_misfit) / abs(previous_misfit)) * 100.0
        improvement_pct_list.append(improvement_pct)
    return improvement_pct_list


def pygmt_begin():
    fig = pygmt.Figure()
    pygmt.config(
        FONT_LABEL="16p",
        FONT_ANNOT_PRIMARY="15p",
    )

    return fig


def plot_misfit(fig, model_beg, model_end, misfit_list):
    model_list = np.arange(model_beg, model_end + 1)
    y_min = float(np.min(misfit_list))
    y_max = float(np.max(misfit_list))
    if np.isclose(y_min, y_max):
        y_pad = max(abs(y_min) * 0.05, 0.01)
    else:
        y_pad = (y_max - y_min) * 0.1
    fig.basemap(
        region=[model_beg - 0.9, model_end + 0.9, y_min - y_pad, y_max + y_pad],
        projection="X8c/10c",
        frame=["WSne+tMisfit", "x1a1f+lModel Number", "y+lWeighted Average Misfit"],
    )
    fig.plot(
        x=model_list,
        y=misfit_list,
        pen="1p,black",
    )
    fig.plot(
        x=model_list,
        y=misfit_list,
        style="c0.2c",
        pen="0.5p,black",
        fill="red",
    )

    return fig


def check_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    return output_dir


if __name__ == "__main__":
    # -----------------PARAMETERS----------------- #
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    adjointflows_dir = f"{base_dir}/adjointflows"
    tomo_dir = f"{base_dir}/TOMO"
    out_dir = f"{tomo_dir}/OUTPUT"
    model_n_list = [26, 28]

    # -------------------------------------------- #

    dataset_config = load_dataset_config(adjointflows_dir)
    dataset_entries = _build_dataset_entries(dataset_config)
    if not dataset_entries:
        raise ValueError("No datasets defined in dataset.yaml; misfit calculation requires dataset entries.")
    default_evlst_name = get_by_path(dataset_config, "defaults.list.evlst")

    misfit_list = get_misfit_list(
        model_beg=model_n_list[0],
        model_end=model_n_list[1],
        dataset_entries=dataset_entries,
        base_dir=base_dir,
        default_evlst_name=default_evlst_name,
    )
    misfit_arr = np.array(misfit_list, dtype=float)
    improvement_pct_list = get_improvement_pct_list(misfit_arr.tolist())

    print("Weighted average misfit by model:")
    for model_num, misfit in zip(range(model_n_list[0], model_n_list[1] + 1), misfit_arr):
        print(f"m{model_num:03d}: {misfit:.5f}")

    print("\nMisfit improvement relative to previous model:")
    for model_num, improvement_pct in zip(
        range(model_n_list[0], model_n_list[1] + 1),
        improvement_pct_list,
    ):
        if improvement_pct is None:
            print(f"m{model_num:03d}: N/A")
        else:
            print(f"m{model_num:03d}: {improvement_pct:.3f}%")

    fig = pygmt_begin()
    fig = plot_misfit(fig=fig, model_beg=model_n_list[0], model_end=model_n_list[1], misfit_list=misfit_arr)

    fig.show()
# %%
# 
