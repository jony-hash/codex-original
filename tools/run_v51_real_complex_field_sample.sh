#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/sample_twophoton_100x100_v51_real_complex_field_probe_a_s025.yaml"
CHECKPOINT="outputs/20260710_v47/runs/20260710_050845_365005_train_v47_probe_a_s025_phase_amplitude_decoder_output_e12/checkpoints/one_shot_distill_best.pt"

test -f "$CONFIG"
test -f "$CHECKPOINT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate optical-generative-models

python train.py \
  --train-config "$CONFIG" \
  --stage one_shot_sample \
  --device cuda \
  --one-shot-checkpoint "$CHECKPOINT" \
  --run-name v51_real_complex_field_probe_a_s025_sample
