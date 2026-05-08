# Final Acceptance Record

This record pins the current accepted optical LDM checkpoint and the GPU sample run used for acceptance.

## Accepted Checkpoint

- Diffusion checkpoint: `outputs/runs/20260507_011608_845740_train_diffusion_grid32_ols1ch_artifact_w055_s20k_b128_w4_wc4x300_g45_e30_weak345n16/checkpoints/diffusion_best.pt`
- First-stage checkpoint: `outputs/runs/20260504_134132_767348_train_first_stage_grid32_ols1ch_artifact_w055_e8/checkpoints/first_stage_best_optical.pt`
- Acceptance sample run: `outputs/runs/20260507_081830_752705_train_sample_grid32_ols1ch_artifact_w055_cw4x300_all_g45_n4_nn_metrics`
- Final copied checkpoint folder: `outputs/final_pt/`
  - `outputs/final_pt/first_stage_best_optical.pt`
  - `outputs/final_pt/diffusion_best.pt`

## Acceptance Metrics

- Device: `cuda`
- `overall_nearest_dice`: `0.8503`
- `overall_nearest_corr`: `0.8662`
- `min_class_nearest_dice`: `0.7857`
- `class4_nearest_dice`: `0.8126`

Acceptance threshold:

- `run_meta.device == "cuda"`
- `overall_nearest_dice >= 0.80`
- `min_class_nearest_dice >= 0.75`

## Latest CUDA Recheck

The accepted checkpoints were rechecked after the visualization fix with the same full-class sample protocol:

- Recheck run: `outputs/runs/20260508_144922_456260_train_sample_grid32_ols1ch_artifact_w055_cw4x300_all_g45_n4_diffusion_visual_fix_recheck`
- Device: `cuda`
- `post_train_sample_config.class_labels`: `0-9`, repeated 4 times per class
- `overall_nearest_dice`: `0.8492`
- `overall_nearest_corr`: `0.8675`
- `min_class_nearest_dice`: `0.7815`

This recheck confirms the final checkpoint still passes the acceptance thresholds after code refactoring and the diffusion visualization fix.

## Diffusion Visualization Smoke

The diffusion training epoch visualization was separately smoke-tested so that generated training-effect figures cover all classes instead of the last random batch subset:

- Smoke run: `outputs/runs/20260508_145449_512992_train_smoke_diffusion_visual_0_9_check`
- Command shape: `--stage full --device cuda --first-stage-epochs 1 --diffusion-epochs 1`
- `configs/diffusion_visual_config.json` records `class_labels == [0,1,2,3,4,5,6,7,8,9]`
- `configs/diffusion_visual_config.json` records `reference_labels == [0,1,2,3,4,5,6,7,8,9]`
- Generated diffusion epoch figure: `images/diffusion/epoch_0001.png`

This smoke run is only a visualization-path check. Its one-epoch checkpoints are not final model checkpoints and must not replace `outputs/final_pt/`.

## Reproduction Command

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate optical-generative-models

python train.py \
  --train-config configs/train_first_stage_linear_metasurface_2000_grid32.yaml \
  --stage sample \
  --device cuda \
  --first-stage-checkpoint outputs/runs/20260504_134132_767348_train_first_stage_grid32_ols1ch_artifact_w055_e8/checkpoints/first_stage_best_optical.pt \
  --diffusion-checkpoint outputs/runs/20260507_011608_845740_train_diffusion_grid32_ols1ch_artifact_w055_s20k_b128_w4_wc4x300_g45_e30_weak345n16/checkpoints/diffusion_best.pt \
  --guidance-scale 4.5 \
  --sample-class-labels 0 1 2 3 4 5 6 7 8 9 \
  --sample-repeats-per-class 4 \
  --run-name sample_grid32_ols1ch_artifact_w055_cw4x300_all_g45_n4_nn_metrics
```

## Notes

The fixed-reference Dice metrics are kept for continuity, but final acceptance uses `nearest_neighbor` metrics because generated handwritten digits can be valid while not matching the one fixed reference image for a class.
