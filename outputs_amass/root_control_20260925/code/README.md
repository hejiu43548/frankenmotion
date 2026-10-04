# FrankenMotion root-control adapter experiment (2026-09-25)

Stage 1: finetune a numeric control branch on top of the user's final 6000-epoch checkpoint. The original model weights are frozen and never overwritten. This is an experimental adapter, not an established-quality model.

## Run

Remote experiment: `/home/pku/frankenmotion/outputs_amass/root_control_20260925`

- Source: `/home/pku/frankenmotion/outputs_amass/official_20260916/logs/checkpoints/last.ckpt`
- Source SHA256: `1de95780b218852af164737201afde6b6878cd7e5407e97538b06588f40b6eac`
- Train/validation: existing available-only splits (12730 / 1634); unavailable AMASS files remain excluded.
- 100 epochs maximum; stop after 15 validation epochs without sufficient improvement.
- Batch size 16, gradient accumulation 4, AdamW learning rate 0.0001.
- 593536 trainable adapter parameters; source Transformer and normalizers frozen.
- Adapter uses a numeric encoder and zero-initialized residual projection after each of four Transformer blocks. This is a lightweight experimental design inspired by control branches, not a reproduction of a full ControlNet paper.
- Input/output feature dimensions remain 613. No modified text or numeric prediction channels.
- Text dropout is disabled in this experiment to avoid upstream in-place caption-cache mutation. Numeric channels are independently omitted with probability 0.15.
- Uses existing source normalization statistics, not statistics fitted to validation data.

## Labels and objective

Automatically derive physical speed (m/s) and signed BODY yaw rate (rad/s) from unnormalized root increments at 20 FPS. Labels are averages within nonoverlapping 1-second windows. Exclude each sequence's last increment. Body yaw is not necessarily trajectory direction, especially for backward/sideways motion. Absolute yaw and start location remain canonical.

Input scales: speed / 3 m/s and yaw rate / pi rad/s, plus availability bits. For a requested constant turn of theta over duration D, use yaw rate theta/D. This is not a turning-radius or path-tangent guarantee.

Loss: valid-frame normalized motion reconstruction (root channels weighted 5), plus 0.1 * speed MSE + 0.1 * yaw-rate MSE on 1-second windows. No explicit contact/physics loss yet. Data has many slow/static motions; unsupported high-speed combinations are not guaranteed. All available actions participate in training, not only running.

## Status

```bash
ssh pku@100.94.2.67 'bash /home/pku/frankenmotion/outputs_amass/root_control_20260925/code/status.sh'
ssh pku@100.94.2.67 'tail -f /home/pku/frankenmotion/outputs_amass/root_control_20260925/train.log'
```

Success requires `status.json` state `completed`, `DONE` present, and supervisor `exit.json` exit_code 0. A stopped process without DONE is NOT success. States: initializing, training, completed, failed.

Artifacts:
- `last.pt`: last epoch adapter, optimizer, RNG states; resumable.
- `best.pt`: best validation objective adapter. Requires original source checkpoint, `base_config.yaml`, and supplied code; it is NOT a drop-in checkpoint for unmodified generate_part.py.
- `metrics.jsonl`: per-epoch training/validation metrics and elapsed time.
- `labels_train.jsonl`, `labels_val.jsonl`, `data_audit.json`: extracted scalar labels and dataset summary; time-varying labels are calculated online with exactly the same rule.
- `provenance.json`: source hash, hyperparameters and adapter size.
- `samples/before`, `samples/after_best`: fixed-seed 50-step deterministic DDIM motion arrays and root-control metrics for 12 held-out annotations. Four annotations also get counterfactual +/-20% speed and +/-0.15 rad/s yaw targets. Baseline branch uses the same frozen user-trained source. This is not an official-weight benchmark.

Teacher-forced denoising validation errors do not establish free-generation control quality. Inspect before/after samples, counterfactual response, foot sliding, semantic quality and diversity before proceeding to arm-frequency training. Samples are SMPL-RIFKE feature arrays, not videos yet.

## Numerical inference after training

Uses a held-out annotation's TEXT, fresh diffusion noise, and explicit requested numeric controls. Ground-truth motion is not fed into denoising motion channels.

```bash
cd /home/pku/frankenmotion
.conda/bin/python outputs_amass/root_control_20260925/code/infer.py \
  --run /home/pku/frankenmotion/outputs_amass/root_control_20260925 \
  --index 0 --speed 0.5 --turn-deg 45 \
  --output /home/pku/frankenmotion/outputs_amass/root_control_20260925/custom/sample.npy
```

The selected annotation may conflict with the requested controls; choose an appropriate locomotion annotation. Outputs include a JSON describing requested targets, not achieved accuracy.

## Resume after an interruption

Only run when the existing training process is stopped:

```bash
cd /home/pku/frankenmotion
nohup .conda/bin/python -u outputs_amass/root_control_20260925/code/launch.py \
 /home/pku/frankenmotion/outputs_amass/root_control_20260925 --epochs 100 --resume \
 > outputs_amass/root_control_20260925/launcher_resume.log 2>&1 < /dev/null &
```

200 GiB `/swapfile-frankenmotion` was verified active and present in `/etc/fstab`. Swap extends CPU RAM, not GPU VRAM.
