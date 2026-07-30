"""Static lazy facade for historical v2 candidate runtime imports."""

from __future__ import annotations

from importlib import import_module


_SYMBOL_MODULES = {
    **{name: "..diagnostic.runtime" for name in (
        "ConditionalBundle", "_require_conditional", "_require_identity",
        "_require_class_modes", "_require_parity", "_require_multi_anchor",
        "_require_current_acceptance", "_system_signature", "_context",
        "_write_evidence", "_new_run", "_split", "_carrier_logits",
        "_random_codes", "_load_bundle", "_load_parity_source",
    )},
    **{name: "..diagnostic.admission" for name in (
        "train_evaluator", "qualify_evaluator", "conditional_injection_admission",
        "conditional_observability", "conditional_control_isolation",
        "sample_conditional",
        "_identity_semantic_localization", "_first_hundred_schedule_hash",
        "_schedule_hash", "_fixed_control_probe",
        "_identity_isolation_classification", "_semantic_localization_classification",
    )},
    **{name: "..diagnostic.recovery" for name in (
        "controllability_smoke", "semantic_recovery_smoke", "conditional_train",
        "conditional_training_completion",
    )},
    **{name: "..diagnostic.diversity" for name in (
        "conditional_diversity_observability", "conditional_diversity_alignment_admission",
        "conditional_identity_reanchor", "conditional_reanchor_observability",
        "identity_reanchor_smoke", "train_identity_reanchor", "_reanchor_codes",
    )},
    **{name: "..diagnostic.class_modes" for name in (
        "conditional_shared_mode_conflict", "conditional_class_mode_admission",
        "conditional_class_mode_observability", "conditional_semantic_tangent",
    )},
    **{name: "..diagnostic.parity" for name in (
        "conditional_parity_conflict", "conditional_parity_mode_admission",
        "conditional_parity_observability", "parity_stabilizer_smoke",
        "train_parity_stabilizer", "class_mode_recovery_smoke", "train_class_modes",
    )},
    **{name: "..diagnostic.legacy_training" for name in (
        "diversity_recovery_smoke", "diversity_aligned_smoke",
        "train_diversity_recovery", "train_diversity_aligned",
    )},
    **{name: "..diagnostic.evaluation" for name in (
        "conditional_evaluate", "admit_conditional_evaluation",
        "export_conditional_dataset", "audit_memorization", "evaluate_utility",
        "admit_dataset",
    )},
    **{name: "..diagnostic.multi_anchor" for name in (
        "conditional_multi_anchor_capacity", "conditional_multi_anchor_admission",
        "conditional_multi_anchor_observability", "multi_anchor_smoke",
        "train_multi_anchor",
    )},
}


def __getattr__(name: str):
    try:
        module_name = _SYMBOL_MODULES[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __package__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_SYMBOL_MODULES))
