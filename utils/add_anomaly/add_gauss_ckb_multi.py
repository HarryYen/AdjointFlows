#%%
from modules_anomaly import *
from multiprocessing import Pool
import numpy as np
import os


# ------------------------------------------------------------
# Utility: convert angular spacing (deg) to meters
# ------------------------------------------------------------
def degree_to_meter(dlon_deg, dlat_deg, lat0_deg):
    """
    Convert longitudinal / latitudinal spacing in degrees
    into approximate meter spacing at a reference latitude.
    """
    R = 6371000.0  # Earth radius (m)
    lat0_rad = np.deg2rad(lat0_deg)

    dy = dlat_deg * np.pi / 180.0 * R
    dx = dlon_deg * np.pi / 180.0 * R * np.cos(lat0_rad)
    return dx, dy


# ------------------------------------------------------------
# Worker function: process one SPECFEM rank
# This function will be executed in parallel by multiprocessing
# ------------------------------------------------------------
def process_one_rank(rank):
    """
    Apply Gaussian checkerboard perturbation to one SPECFEM rank.

    Notes
    -----
    - All heavy geometry (centers, grids) is prepared in the main process.
    - This function only reads local data for the given rank.
    - Large arrays are accessed as read-only global variables
      (efficient under fork / copy-on-write).
    """
    print(f"[PID {os.getpid()}] processing rank {rank}")

    # Input model file for this rank
    vector_file = os.path.join(databases_dir, f"proc{rank:06d}_{vector}.bin")

    # Read model vector
    vector_arr, vector_data_type = read_binary_float32(vector_file)

    # Coordinates for this rank
    x_arr = x_arr_list[rank]
    y_arr = y_arr_list[rank]
    z_arr = z_arr_list[rank]

    # Convert UTM coordinates back to lon/lat for block selection
    lon_arr, lat_arr = utm_to_lonlat(
        x=x_arr,
        y=y_arr,
        utm_zone=utm_zone,
        is_north_hemisphere=is_north_hemisphere,
    )

    # Build Gaussian checkerboard perturbation
    ckb_pert = build_gaussian_ckb(
        lon_arr=lon_arr,
        lat_arr=lat_arr,
        x_arr=x_arr,
        y_arr=y_arr,
        z_arr=z_arr,
        lon_centers=lon_centers,
        lat_centers=lat_centers,
        z_centers=z_centers,
        x_grid=x_grid,
        y_grid=y_grid,
        dlon_deg=dlon_deg,
        dlat_deg=dlat_deg,
        utm_zone=utm_zone,
        is_north_hemisphere=is_north_hemisphere,
        dz=dz_meter,
        sigma=(sx, sy, sz),
        n_sigma=2,
        amplitude=amplitude_gaussian,
    )
    print("ckb_pert absmax:", np.max(np.abs(ckb_pert)))
    print("ckb_pert min/max:", ckb_pert.min(), ckb_pert.max())

    # Apply perturbation multiplicatively
    vector_arr *= (1.0 + ckb_pert)

    # Write output
    outfile = os.path.join(output_dir, os.path.basename(vector_file))
    output_binary_file(
        data=vector_arr,
        outfile=outfile,
        data_type_int=ibool_data_type,
        data_type_float=vector_data_type,
    )

    return rank


# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------
if __name__ == "__main__":

    # --------------------
    # Parameters
    # --------------------
    databases_dir = "DATABASES_MPI"
    vector = "vs"
    nproc = 2
    output_dir = "OUTPUT"

    lon_range_deg = [119, 123]
    lat_range_deg = [21, 26]
    dep_range_meter = [0, 0]  # downward positive
    dlon_deg, dlat_deg = 0.3, 0.3
    dz_meter = 20000.0
    lat0_deg = 24.0
    edge_amplitude_ratio_gaussian = 0.2
    amplitude_gaussian = 0.01
    utm_zone = 50
    is_north_hemisphere = True

    # --------------------
    # Convert angular spacing to meters and suggest Gaussian sigma
    # --------------------
    dx_meter, dy_meter = degree_to_meter(dlon_deg, dlat_deg, lat0_deg)

    sx = suggest_sigma_from_dx(edge_amplitude_ratio_gaussian, dx_meter)
    sy = suggest_sigma_from_dx(edge_amplitude_ratio_gaussian, dy_meter)
    sz = suggest_sigma_from_dx(edge_amplitude_ratio_gaussian, dz_meter)
    print(f'The suggested Gaussian sigma (m) is: sx={sx:.1f}, sy={sy:.1f}, sz={sz:.1f}')
    # --------------------
    # Expand domain by one cell to avoid truncation at boundaries
    # --------------------
    lon_start = lon_range_deg[0] - dlon_deg
    lon_end   = lon_range_deg[1] + dlon_deg
    lat_start = lat_range_deg[0] - dlat_deg
    lat_end   = lat_range_deg[1] + dlat_deg
    # dep_start = dep_range_meter[0] - dz_meter
    dep_start = dep_range_meter[0]
    dep_end   = dep_range_meter[1] + dz_meter

    nx = int(np.ceil((lon_end - lon_start) / dlon_deg))
    ny = int(np.ceil((lat_end - lat_start) / dlat_deg))
    nz = int(np.ceil((dep_end - dep_start) / dz_meter))
    
    # tell user the expanded grid info
    print(f'Expanded grid info:')
    print(f'  lon: {lon_start:.3f} to {lon_end:.3f} deg, nx={nx}')
    print(f'  lat: {lat_start:.3f} to {lat_end:.3f} deg, ny={ny}')
    print(f'  dep: {dep_start:.1f} to {dep_end:.1f} m, nz={nz}')
    # --------------------

    lon_centers = lon_start + (np.arange(nx) + 0.5) * dlon_deg
    lat_centers = lat_start + (np.arange(ny) + 0.5) * dlat_deg
    z_centers   = -1.0 * (dep_start + (np.arange(nz) + 0.5) * dz_meter)

    # --------------------
    # Project lon/lat checkerboard centers to UTM
    # --------------------
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers, indexing="ij")
    x_grid, y_grid = lonlat_to_utm(
        lon_grid,
        lat_grid,
        utm_zone=utm_zone,
        is_north_hemisphere=is_north_hemisphere,
    )

    # --------------------
    # Read coordinates for all ranks (once)
    # --------------------
    x_arr_list, y_arr_list, z_arr_list = [], [], []
    ibool_arr_list = []
    ibool_data_type = None

    for rank in range(nproc):
        ibool_file = os.path.join(databases_dir, f"proc{rank:06d}_ibool.bin")
        x_file     = os.path.join(databases_dir, f"proc{rank:06d}_x.bin")
        y_file     = os.path.join(databases_dir, f"proc{rank:06d}_y.bin")
        z_file     = os.path.join(databases_dir, f"proc{rank:06d}_z.bin")

        ibool_arr, ibool_data_type_tmp = read_binary_int32(ibool_file)
        x_raw, _ = read_binary_float32(x_file)
        y_raw, _ = read_binary_float32(y_file)
        z_raw, _ = read_binary_float32(z_file)

        x_arr = x_raw[ibool_arr - 1]
        y_arr = y_raw[ibool_arr - 1]
        z_arr = z_raw[ibool_arr - 1]

        x_arr_list.append(x_arr)
        y_arr_list.append(y_arr)
        z_arr_list.append(z_arr)
        ibool_arr_list.append(ibool_arr)

        if ibool_data_type is None:
            ibool_data_type = ibool_data_type_tmp

    os.makedirs(output_dir, exist_ok=True)

    # --------------------
    # Parallel execution: one process per rank
    # --------------------
    with Pool(processes=nproc) as pool:
        pool.map(process_one_rank, range(nproc))

# %%