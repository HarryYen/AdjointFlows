# Checkerboard Synthetic Test Flow

This repository already has most of the pieces needed for a checkerboard test.
What is missing is a small wrapper flow that turns synthetics generated from a
perturbed model into a fake observed dataset that the normal inversion pipeline
can consume.

The recommended flow is:

## 1. Build a perturbed "true" model

Create a perturbed `DATABASES_MPI` directory with one of the existing tools:

- `utils/add_anomaly/add_gauss_ckb.py`
- `utils/add_anomaly/add_gauss_ckb_multi.py`
- `utils/create_synmodel/add_ckb.py`
- `utils/create_synmodel/create_layered_model_for_specfem.py`

Suggested practice:

- Copy a reference model directory to a dedicated model number such as `m900`.
- Put the checkerboard-perturbed binaries in `TOMO/m900/DATABASES_MPI`.
- Keep the inversion starting model separate, for example `m000`.

## 2. Generate synthetic waveforms from the perturbed model

Run the existing workflow in forward-only mode, using the perturbed model as
the simulation model.

Recommended settings in `adjointflows/config.yaml`:

```yaml
setup:
  workflow:
    inversion_execution_mode: single
    start_from_stage: forward
    end_at_stage: forward
    forward_stop_at: synthetics
  model:
    current_model_num: 900
    do_generate: 0
    do_mesh: 0
```

Recommended dataset settings:

- Keep the same `list.evlst` and `list.stlst` you want to use in the inversion.
- Set `synthetics.do_wave_simulation: 1`.
- Point `synthetics.waveform_dir` to a dedicated folder such as `SYN_EQ_TRUE`.
- `data.waveform_dir` is not used when `forward_stop_at: synthetics`, so it can
  stay as-is.

Then run:

```bash
cd adjointflows
python main.py
```

After this step you should have something like:

```text
TOMO/m900/SYN_EQ_TRUE/<event>/
```

with files such as:

```text
TW.STA.BXE.semv.convolved.sac
TW.STA.BXN.semv.convolved.sac
TW.STA.BXZ.semv.convolved.sac
```

## 3. Repackage those synthetics as fake observed data

Use the helper script added in this repository:

1. Open [prepare_checkerboard_data.py](/home/harry/Work/adjflows_for_ambient_noise/AdjointFlows/scripts/prepare_checkerboard_data.py).
2. Edit the `DEFAULT_CONFIG` block near the top of the file.
3. Run:

```bash
python scripts/prepare_checkerboard_data.py
```

Recommended values:

```python
DEFAULT_CONFIG = {
    "source_syn_dir": "TOMO/m900/SYN_EQ_TRUE",
    "event_list": "DATA/evlst/fwi_new_cat_version4.txt",
    "output_data_dir": "DATA/wav_EQ_checkerboard",
    "template_data_dir": "DATA/wav_EQ",
    "station_list": "DATA/stlst/sta_new_remove_western.txt",
    "network": "TW",
    "synthetic_component": "semv",
    "clean_output": True,
    "strict": False,
    "dry_run": False,
}
```

What this does:

- Reads the event list and synthetic event folders.
- Copies `TW.STA.BX?.semv.convolved.sac` into `DATA/wav_EQ_checkerboard/<event>/`.
- Reuses the exact filename from `DATA/wav_EQ/<event>/` when a matching station
  and component already exist there.
- Falls back to the standard `STA.HH?.YYYY.JJJ.HH.MM.sac` naming pattern when
  no template filename exists.

This makes the synthetic waveforms look like normal observed data to the
existing `flexwin/ini_proc.bash` preprocessing step.

If you ever want temporary overrides, the script still supports CLI arguments,
but editing `DEFAULT_CONFIG` is now the main workflow.

## 4. Run the normal inversion against the fake observed dataset

Switch back to the model you want to invert from, for example `m000`, and use
the checkerboard data directory as the observed data.

Recommended dataset settings:

```yaml
- name: EQ_5_12s
  data:
    waveform_dir: wav_EQ_checkerboard
  synthetics:
    do_wave_simulation: 1
    waveform_dir: SYN_EQ
  list:
    evlst: fwi_new_cat_version4.txt
    stlst: sta_new_remove_western.txt
```

Recommended workflow settings:

```yaml
setup:
  workflow:
    inversion_execution_mode: continuous
  model:
    current_model_num: 0
```

Now the workflow becomes:

1. observed data = fake checkerboard waveforms in `DATA/wav_EQ_checkerboard`
2. synthetic prediction = waveforms from the current inversion model
3. measurement/adjoint/kernel/update = unchanged normal pipeline

## 5. Summary of the full checkerboard workflow

The full logic is:

1. build a perturbed "true" model
2. generate waveforms from that true model
3. copy those waveforms into a `DATA/wav_*` style tree
4. run the standard inversion starting from the reference model
5. compare the recovered model with the known checkerboard input

## Notes

- The key design choice is to avoid modifying the main inversion workflow.
- `flexwin` already expects `DATA/wav/<event>` and `SYN/<event>` after symlink
  setup, so only the fake observed packaging step is new.
- If you want multiple checkerboard tests, keep one observed directory per test,
  for example `wav_EQ_ckb_50km`, `wav_EQ_ckb_80km`, or `wav_EQ_ckb_gauss01`.
