from __future__ import annotations

from ..sampling_reports import save_post_train_samples
from ..train_utils import (
    load_latent_scale_from_diffusion_checkpoint,
    set_first_stage_image_fusion_weight,
)


def run_sample_stage(
    torch_module,
    *,
    first_stage,
    diffusion_model,
    scheduler,
    train_bundle,
    validation_bundle,
    train_cfg,
    run_manager,
    run_meta: dict[str, object],
    device,
    diffusion_checkpoint: str,
    first_stage_model_version: str,
    load_diffusion_checkpoint_fn,
    sample_latents_fn,
    sample_latents_with_trace_fn,
    resolve_denoising_trace_steps_fn,
) -> None:
    first_stage.eval()
    set_first_stage_image_fusion_weight(
        first_stage, train_cfg.first_stage.image_fusion_end_weight
    )
    diffusion_payload = torch_module.load(
        diffusion_checkpoint, map_location=device, weights_only=False
    )
    load_diffusion_checkpoint_fn(
        diffusion_model, diffusion_payload, first_stage_model_version
    )
    latent_scale = load_latent_scale_from_diffusion_checkpoint(
        torch_module, diffusion_payload, device
    )
    diffusion_model.eval()
    run_meta.update(
        {
            "latent_scale_source": "diffusion_checkpoint",
            "diffusion_best_loaded_for_sampling": True,
        }
    )
    run_manager.write_meta(run_meta)
    print(f"loaded diffusion checkpoint for sampling: {diffusion_checkpoint}")
    sample_artifacts = save_post_train_samples(
        torch_module,
        run_manager,
        first_stage,
        diffusion_model,
        scheduler,
        train_bundle,
        validation_bundle,
        train_cfg,
        latent_scale,
        device,
        sample_latents_fn,
        sample_latents_with_trace_fn,
        resolve_denoising_trace_steps_fn,
    )
    run_meta.update(sample_artifacts)
    run_manager.write_meta(run_meta)
