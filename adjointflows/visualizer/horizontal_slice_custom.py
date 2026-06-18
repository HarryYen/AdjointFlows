#%%
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pygmt

from plotting_modules import (
    find_minmax_from_xyz_file,
    find_nxnynz_from_xyz_file,
    interp_2d_in_specific_dep,
)


# =============================================================================
# User settings: edit this block directly, then run this file.
# =============================================================================
MODEL_NUM = 30
TOMO_DIR = Path("/home/harry/Work/adjflows_for_ambient_noise/AdjointFlows/TOMO")

# Choose one: "vp", "vs", "rho", "vpvs", "dvp", "dvs", "drho".
SCALAR = "vp"
DEPTH_KM = 15

MAP_REGION = [119.0, 123.0, 21.0, 26.0]  # [lon_min, lon_max, lat_min, lat_max]

# Set to None for automatic color range from the selected depth slice.
# Otherwise use [min, max, interval], for example [4.0, 7.0, 0.01].
CPT_RANGE = [5.0, 7.0, 0.1]
COLORBAR_INTERVAL = 0.5
COLORBAR_POSITION = 'JBC+w3.5c/0.3c+v+o5.5c/-4.5c'

CMAP = "roma"
REVERSE_CMAP = False
FINE_GRID_SPACING = 0.01
PROJECTION = "M10c"
DPI = 300

OUTPUT_DIR = TOMO_DIR / f"m{MODEL_NUM:03d}" / "OUTPUT" / "fig" / "horizontal_slice_custom"
OUTPUT_NAME = f"{SCALAR}_{DEPTH_KM:g}km.png"


SCALAR_META = {
    "vp": {"column": 3, "unit": "km/s", "label": "Vp"},
    "vs": {"column": 4, "unit": "km/s", "label": "Vs"},
    "rho": {"column": 5, "unit": "g/cm^3", "label": "Density"},
    "vpvs": {"unit": "", "label": "Vp/Vs"},
    "dvp": {"column": 6, "unit": "%", "label": "dVp"},
    "dvs": {"column": 7, "unit": "%", "label": "dVs"},
    "drho": {"column": 8, "unit": "%", "label": "dDensity"},
}


def load_model_array(input_file, scalar):
    nx, ny, nz = find_nxnynz_from_xyz_file(input_file)
    all_arr_flat = np.loadtxt(input_file, skiprows=5)

    lon_arr = all_arr_flat[:, 0]
    lat_arr = all_arr_flat[:, 1]
    dep_arr = all_arr_flat[:, 2]
    lon_uniq = np.unique(lon_arr)
    lat_uniq = np.unique(lat_arr)
    dep_uniq = np.unique(dep_arr)

    if scalar == "vpvs":
        scalar_flat = all_arr_flat[:, 3] / all_arr_flat[:, 4]
    else:
        column = SCALAR_META[scalar]["column"]
        if column >= all_arr_flat.shape[1]:
            raise ValueError(
                f"{input_file} has {all_arr_flat.shape[1]} columns, "
                f"so scalar '{scalar}' is not available."
            )
        scalar_flat = all_arr_flat[:, column]

    scalar_arr = scalar_flat.reshape(nz, nx, ny)
    return nx, ny, lon_uniq, lat_uniq, dep_uniq, scalar_arr


def make_auto_cpt_range(selected_df, n_interval=100):
    vmin = float(selected_df["scalar"].min())
    vmax = float(selected_df["scalar"].max())
    if np.isclose(vmin, vmax):
        padding = abs(vmin) * 0.05 if vmin else 1.0
        vmin -= padding
        vmax += padding
    interval = (vmax - vmin) / n_interval
    return [vmin, vmax, interval]


def make_colorbar_frame(cbar_label):
    if COLORBAR_INTERVAL is None:
        return f"af+l{cbar_label}"
    return f"a{COLORBAR_INTERVAL:g}f{COLORBAR_INTERVAL:g}+l{cbar_label}"


def plot_horizontal_slice():
    scalar = SCALAR.lower()
    if scalar not in SCALAR_META:
        raise ValueError(f"SCALAR must be one of {list(SCALAR_META)}")

    input_dir = TOMO_DIR / f"m{MODEL_NUM:03d}" / "OUTPUT"
    input_file = input_dir / "model.xyz"
    _, _, _, _, dep_min, dep_max = find_minmax_from_xyz_file(input_file)
    if not dep_min <= DEPTH_KM <= dep_max:
        raise ValueError(f"DEPTH_KM={DEPTH_KM} is outside model depth range [{dep_min}, {dep_max}]")

    nx, ny, lon_uniq, lat_uniq, dep_uniq, scalar_arr = load_model_array(input_file, scalar)
    selected_df = interp_2d_in_specific_dep(lon_uniq, lat_uniq, dep_uniq, scalar_arr, DEPTH_KM)

    cpt_range = CPT_RANGE if CPT_RANGE is not None else make_auto_cpt_range(selected_df)
    grd_region = [lon_uniq.min(), lon_uniq.max(), lat_uniq.min(), lat_uniq.max()]
    meta = SCALAR_META[scalar]
    cbar_label = meta["label"] if not meta["unit"] else f'{meta["label"]} ({meta["unit"]})'

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / OUTPUT_NAME

    fig = pygmt.Figure()
    pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain")

    with TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        coarse_grid = tmp_dir / "slice.grd"
        fine_grid = tmp_dir / "slice_fine.grd"

        pygmt.xyz2grd(
            data=selected_df,
            outgrid=str(coarse_grid),
            region=grd_region,
            spacing=f"{nx}+n/{ny}+n",
            verbose="q",
        )
        pygmt.grdsample(
            grid=str(coarse_grid),
            spacing=FINE_GRID_SPACING,
            region=grd_region,
            outgrid=str(fine_grid),
            verbose="q",
        )

        pygmt.makecpt(cmap=CMAP, series=cpt_range, reverse=REVERSE_CMAP)
        fig.grdimage(grid=str(fine_grid), cmap=True, region=MAP_REGION, projection=PROJECTION, frame=True)
        fig.coast(shorelines=True)
        fig.text(text=f"dep: {DEPTH_KM:g} km", font="20p,Helvetica-Bold", position="BR", frame=True)
        pygmt.config(FONT_ANNOT_PRIMARY="20p,Helvetica")
        pygmt.config(FONT_LABEL="20p,Helvetica")
        fig.colorbar(frame=make_colorbar_frame(cbar_label), position=COLORBAR_POSITION)
        # fig.savefig(str(output_file), dpi=DPI, transparent=True)
        fig.show()

    print(f"Saved: {output_file}")


if __name__ == "__main__":
    plot_horizontal_slice()

# %%
