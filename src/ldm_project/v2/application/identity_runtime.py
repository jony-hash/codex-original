"""Static lazy compatibility facade for the formal identity runtime."""

from __future__ import annotations

from importlib import import_module


_SYMBOL_MODULES = {
    **{name: ".current.runtime" for name in (
        "ConditionalBundle", "_require_identity", "_require_conditional",
        "_require_current_acceptance", "_system_signature", "_context",
        "_write_evidence", "_new_run", "_split", "_carrier_logits",
        "_random_codes", "_load_bundle",
    )},
    **{name: ".current.admission" for name in (
        "train_evaluator", "qualify_evaluator", "conditional_injection_admission",
        "conditional_observability", "conditional_control_isolation",
        "_identity_semantic_localization", "_conditional_parameter_groups",
    )},
    **{name: ".current.recovery" for name in (
        "controllability_smoke", "semantic_recovery_smoke",
    )},
    **{name: ".current.training" for name in (
        "conditional_train", "conditional_training_completion",
    )},
    **{name: ".current.evaluation" for name in (
        "sample_conditional", "conditional_evaluate", "admit_conditional_evaluation",
    )},
    **{name: ".current.dataset" for name in (
        "export_conditional_dataset", "audit_memorization", "evaluate_utility",
        "admit_dataset",
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
