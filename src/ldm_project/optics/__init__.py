"""Physical optics public API.

The package keeps the historical ``ldm_project.optics`` import path while
separating field provenance, propagation, sampling, devices, relays and
independent references into explicit modules.
"""
from .devices import IdealIdentityElement, ReflectiveAmplitudeSLM
from .fields import (
    PHASE_INTENSITY_FLOOR_RATIO,
    STRICT_OPTICAL_FIELD_SCHEMA_VERSION,
    build_optical_field_record,
    build_strict_optical_field_views,
    detach_optical_fields,
    save_strict_optical_field_grids,
    validate_optical_field_chain,
)
from .propagation import (
    angular_spectrum_propagate,
    apply_thin_lens,
    band_limited_angular_spectrum_propagate,
    fresnel_propagate,
    normalize_intensity,
    normalize_real_intensity,
)
from .relays import (
    BackOpticalPath,
    FrontOpticalPath,
    _gaussian_blur_complex_field,
    _gaussian_blur_intensity,
    coherent_imaging_4f_field,
    ideal_image_field,
)
from .sampling import PlaneSampling, aperture_sampling, sample_center_aperture, validate_sampling_transition

__all__ = [
    "BackOpticalPath",
    "FrontOpticalPath",
    "IdealIdentityElement",
    "PHASE_INTENSITY_FLOOR_RATIO",
    "PlaneSampling",
    "ReflectiveAmplitudeSLM",
    "STRICT_OPTICAL_FIELD_SCHEMA_VERSION",
    "_gaussian_blur_complex_field",
    "_gaussian_blur_intensity",
    "angular_spectrum_propagate",
    "aperture_sampling",
    "apply_thin_lens",
    "band_limited_angular_spectrum_propagate",
    "build_optical_field_record",
    "build_strict_optical_field_views",
    "coherent_imaging_4f_field",
    "detach_optical_fields",
    "fresnel_propagate",
    "ideal_image_field",
    "normalize_intensity",
    "normalize_real_intensity",
    "sample_center_aperture",
    "save_strict_optical_field_grids",
    "validate_optical_field_chain",
    "validate_sampling_transition",
]
