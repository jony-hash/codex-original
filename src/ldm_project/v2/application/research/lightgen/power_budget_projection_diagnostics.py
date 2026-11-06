"""Independent power accounting; LS coefficients never enter an optical producer."""
from __future__ import annotations

import math

import torch


def area_power(field: torch.Tensor, area_m2: float) -> torch.Tensor:
    if not torch.is_complex(field) or not torch.isfinite(field).all() or not math.isfinite(area_m2) or area_m2 <= 0:
        raise ValueError("power requires finite complex samples and positive pixel area")
    return field.to(torch.complex128).abs().square().flatten(1).sum(1)*area_m2


class OLSProjectionPowerAudit:
    """Keep direct overlaps, production synthesis, and LS projection distinct."""

    def __init__(self, modes: torch.Tensor, area_m2: float, gram_rtol: float = 1e-12):
        if (modes.ndim != 2 or not torch.is_complex(modes) or not torch.isfinite(modes).all()
                or not math.isfinite(area_m2) or area_m2 <= 0 or not 0 < gram_rtol < 1):
            raise ValueError("expected finite complex [pixels,modes], positive area and valid Gram cutoff")
        self.v = modes.to(torch.complex128)
        self.area = area_m2
        self.rtol = gram_rtol
        self.gram = self.v.mH@self.v*self.area
        self.eigenvalues = torch.linalg.eigvalsh(self.gram)
        self.cutoff = self.eigenvalues[-1].item()*gram_rtol
        self.rank = int((self.eigenvalues > self.cutoff).sum())
        self.inverse = torch.linalg.pinv(self.gram, hermitian=True, rtol=gram_rtol)

    def basis_diagnostics(self) -> dict:
        diagonal = self.gram.diagonal().real
        if (diagonal <= 0).any():
            raise ValueError("zero-energy modes have no normalized overlap")
        normalized = self.gram/torch.sqrt(diagonal[:, None]*diagonal[None, :])
        mask = ~torch.eye(len(diagonal), device=diagonal.device, dtype=torch.bool)
        overlaps = normalized.abs()[mask]
        condition = (self.eigenvalues[-1]/self.eigenvalues[0]).item() if self.eigenvalues[0] > 0 else None
        spectrum_error = (self.eigenvalues-1).abs().max().item()
        max_overlap = overlaps.max().item() if overlaps.numel() else 0.0
        if self.rank < len(diagonal) or condition is None or condition >= 1/self.rtol:
            status = "OLS_BASIS_ILL_CONDITIONED"
        elif spectrum_error <= 5e-6 and max_overlap <= 5e-6:
            status = "OLS_BASIS_NEAR_ORTHOGONAL"
        else:
            status = "OLS_BASIS_NONORTHOGONAL"
        return {"basis_status":status, "mode_count":len(diagonal),
                "mode_squared_norms":diagonal.cpu().tolist(), "mode_norms":diagonal.sqrt().cpu().tolist(),
                "gram_real":self.gram.real.cpu().tolist(), "gram_imag":self.gram.imag.cpu().tolist(),
                "normalized_gram_real":normalized.real.cpu().tolist(), "normalized_gram_imag":normalized.imag.cpu().tolist(),
                "max_normalized_off_diagonal_overlap":max_overlap,
                "mean_normalized_off_diagonal_overlap":overlaps.mean().item() if overlaps.numel() else 0.,
                "eigenvalues":self.eigenvalues.cpu().tolist(), "numerical_rank":self.rank,
                "condition_number":condition, "gram_pinv_rtol":self.rtol, "gram_absolute_cutoff":self.cutoff,
                "max_mode_squared_norm_error":(diagonal-1).abs().max().item(),
                "gram_spectral_identity_error":spectrum_error,
                "coefficient_definition":"production: direct area-weighted overlaps b; diagnostic LS: G^+ b"}

    def analyze(self, field: torch.Tensor, production_coefficients: torch.Tensor) -> dict[str, torch.Tensor]:
        if (not torch.is_complex(field) or field.ndim != 4 or field.shape[1] != 1
                or field[0].numel() != self.v.shape[0] or not torch.isfinite(field).all()
                or production_coefficients.shape != (field.shape[0], self.v.shape[1])
                or not torch.is_complex(production_coefficients) or not torch.isfinite(production_coefficients).all()):
            raise ValueError("projection audit needs finite same-plane complex fields and production coefficients")
        flat = field.to(torch.complex128).flatten(1)
        z = production_coefficients.to(torch.complex128)
        b = flat@self.v.conj()*self.area
        ls = b@self.inverse.T
        match_field = z@self.v.T
        ls_field = ls@self.v.T
        residual = flat-ls_field
        p_input = flat.abs().square().sum(1)*self.area
        p_match = match_field.abs().square().sum(1)*self.area
        p_ls = ls_field.abs().square().sum(1)*self.area
        p_res = residual.abs().square().sum(1)*self.area
        gram_match = torch.einsum("bi,ij,bj->b", z.conj(), self.gram, z).real
        gram_ls = torch.einsum("bi,ij,bj->b", ls.conj(), self.gram, ls).real
        orth = (residual@self.v.conj()*self.area).abs()/torch.sqrt(
            p_input[:,None].clamp_min(1e-300)*self.gram.diagonal().real[None].clamp_min(1e-300))
        safe = p_input.clamp_min(1e-300)
        return {"P_encoder_w":p_input, "P_coefficient_norm_metric":z.abs().square().sum(1),
                "P_match_w":p_match, "P_match_gram_w":gram_match,
                "P_LS_w":p_ls, "P_LS_gram_w":gram_ls, "P_residual_w":p_res,
                "ratio_coefficient_norm":z.abs().square().sum(1)/safe,
                "ratio_match":p_match/safe, "ratio_LS":p_ls/safe, "ratio_residual":p_res/safe,
                "production_overlap_relative_error":(z-b).norm(dim=1)/b.norm(dim=1).clamp_min(1e-300),
                "match_vs_LS_coefficient_relative":(z-ls).norm(dim=1)/ls.norm(dim=1).clamp_min(1e-300),
                "match_vs_LS_field_relative":(match_field-ls_field).norm(dim=1)/ls_field.norm(dim=1).clamp_min(1e-300),
                "match_gram_closure_relative":(p_match-gram_match).abs()/p_match.clamp_min(1e-300),
                "LS_gram_closure_relative":(p_ls-gram_ls).abs()/p_ls.clamp_min(1e-300),
                "LS_pythagorean_closure_relative":(p_input-p_ls-p_res).abs()/safe,
                "LS_residual_orthogonality":orth.max(dim=1).values,
                "overlaps_b":b, "coefficients_LS":ls,
                "field_match":match_field.reshape(field.shape), "field_LS":ls_field.reshape(field.shape)}


def validate_projection_accounting(result: dict[str, torch.Tensor], *, closure_tolerance=1e-8,
                                   reference_tolerance=1e-10) -> None:
    for name in ("match_gram_closure_relative", "LS_gram_closure_relative", "LS_pythagorean_closure_relative"):
        if result[name].max().item() > closure_tolerance:
            raise ValueError(f"power closure failed: {name}")
    if result["LS_residual_orthogonality"].max().item() > reference_tolerance:
        raise ValueError("LS residual is not orthogonal to the retained basis")
    if (result["P_LS_w"] > result["P_encoder_w"]*(1+closure_tolerance)+1e-15).any():
        raise ValueError("LS captured power exceeds input field power")


def validate_production_handoff(actual: torch.Tensor, production: torch.Tensor) -> None:
    if not torch.equal(actual, production):
        raise ValueError("diagnostic LS/renormalized field cannot replace the production material input")
