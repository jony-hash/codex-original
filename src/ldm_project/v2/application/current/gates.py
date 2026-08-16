"""Single source of truth for current v2 evidence transitions.

The registry describes authorization semantics only.  Scientific metric
calculations remain in their producing stages; consumers use the validators
here to make the rejection decision consistently.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ...evidence import EvidenceContextV2, EvidenceRecordV2


@dataclass(frozen=True)
class CurrentGateContract:
    evidence_kind: str
    purpose: str
    permitted_by_status: Mapping[str, tuple[str, ...]]
    formal: bool = True
    required_context_fields: tuple[str, ...] = ()
    required_input_keys: tuple[str, ...] = ()


def _contract(
    kind: str,
    purpose: str,
    actions: Mapping[str, tuple[str, ...]],
    *,
    required_context_fields: tuple[str, ...] = (),
    required_input_keys: tuple[str, ...] = (),
    formal: bool = True,
) -> CurrentGateContract:
    return CurrentGateContract(
        kind,
        purpose,
        actions,
        formal,
        required_context_fields,
        required_input_keys,
    )


CURRENT_GATE_CONTRACTS: Mapping[str, CurrentGateContract] = {
    "evaluator_qualification": _contract(
        "evaluator_qualification",
        "qualify the frozen primary evaluator on the fixed MNIST split",
        {"passed": ("train_acceptance_evaluator",)},
        required_context_fields=("evaluator_checkpoint_hash", "dataset_split_manifest_hash"),
        required_input_keys=("evaluator_checkpoint",),
    ),
    "acceptance_evaluator_training": _contract(
        "acceptance_evaluator_training",
        "qualify the independent secondary acceptance evaluator",
        {"passed": ("qualify_acceptance_measurement",)},
        required_context_fields=("evaluator_checkpoint_hash", "dataset_split_manifest_hash"),
        required_input_keys=("acceptance_evaluator", "primary_evaluator_qualification"),
    ),
    "conditional_acceptance_measurement": _contract(
        "conditional_acceptance_measurement",
        "bind both evaluators and the official test manifest for acceptance",
        {"passed": ("conditional_injection_admission", "evaluate")},
        required_context_fields=("evaluator_checkpoint_hash", "dataset_split_manifest_hash"),
        required_input_keys=("primary_evaluator", "secondary_evaluator", "measurement_bundle", "test_manifest", "exposure_genesis", "exposure_chain_id"),
    ),
    "conditional_injection_admission": _contract(
        "conditional_injection_admission",
        "prove label and code affect the generator command",
        {"passed": ("conditional_observability",)},
        required_context_fields=("evaluator_checkpoint_hash", "dataset_split_manifest_hash", "checkpoint_hash"),
        required_input_keys=("checkpoint", "measurement_evidence"),
    ),
    "conditional_observability": _contract(
        "conditional_observability",
        "prove control changes reach CMOS through the formal optical path",
        {"passed": ("conditional_control_isolation",)},
        required_context_fields=("evaluator_checkpoint_hash", "dataset_split_manifest_hash", "checkpoint_hash"),
    ),
    "conditional_control_isolation": _contract(
        "conditional_control_isolation",
        "classify whether CMOS change is attributable to the target control",
        {
            "passed": ("conditional_semantic_localization", "controllability_smoke"),
            "diagnostic": ("conditional_semantic_localization", "controllability_smoke"),
            "valid_failed": (),
            "invalid_harness": (),
        },
        required_context_fields=("evaluator_checkpoint_hash", "dataset_split_manifest_hash", "checkpoint_hash"),
    ),
    "conditional_semantic_localization": _contract(
        "conditional_semantic_localization",
        "prove a usable local semantic gradient exists at the target measurement",
        {"passed": ("controllability_smoke",), "valid_failed": (), "invalid_harness": ()},
        required_context_fields=("evaluator_checkpoint_hash", "dataset_split_manifest_hash", "checkpoint_hash"),
    ),
    "controllability_smoke": _contract(
        "controllability_smoke",
        "prove the semantic path is usable within the registered 100-step window",
        {"passed": ("train",), "valid_failed": ("semantic_recovery_smoke",), "invalid_harness": ()},
        required_context_fields=("evaluator_checkpoint_hash", "dataset_split_manifest_hash"),
    ),
    "conditional_semantic_recovery_smoke": _contract(
        "conditional_semantic_recovery_smoke",
        "prove the registered 400-step recovery path",
        {"passed": ("train",), "valid_failed": (), "invalid_harness": ()},
        required_context_fields=("evaluator_checkpoint_hash", "dataset_split_manifest_hash", "checkpoint_hash"),
    ),
    "conditional_training": _contract(
        "conditional_training",
        "produce a finite checkpoint from the fixed formal schedule",
        {"passed": ("evaluate",), "invalid_harness": ()},
        required_context_fields=("evaluator_checkpoint_hash", "dataset_split_manifest_hash", "checkpoint_hash"),
        required_input_keys=("checkpoint", "training_authorization_kind", "training_authorization_evidence_hash"),
    ),
    "conditional_acceptance": _contract(
        "conditional_acceptance",
        "measure the trained optical model with both frozen evaluators",
        {"pending_review": ("admit_conditional_evaluation",), "valid_failed": ()},
        required_context_fields=("evaluator_checkpoint_hash", "dataset_split_manifest_hash", "checkpoint_hash"),
    ),
    "conditional_generation_admission": _contract(
        "conditional_generation_admission",
        "combine automatic evaluation and fixed manual readability review",
        {"passed": ("export",), "valid_failed": ()},
        required_context_fields=("evaluator_checkpoint_hash", "dataset_split_manifest_hash", "checkpoint_hash"),
    ),
    "conditional_dataset_candidate": _contract(
        "conditional_dataset_candidate",
        "write a complete candidate dataset with bound shards",
        {"passed": ("audit_memorization", "evaluate_utility")},
        required_context_fields=("evaluator_checkpoint_hash", "dataset_split_manifest_hash", "checkpoint_hash", "candidate_dataset_manifest_hash"),
    ),
    "conditional_memorization_audit": _contract(
        "conditional_memorization_audit",
        "measure nearest-neighbor memorization of the candidate",
        {"passed": ("admit_dataset",), "valid_failed": ()},
        required_context_fields=("candidate_dataset_manifest_hash",),
    ),
    "conditional_utility": _contract(
        "conditional_utility",
        "measure downstream utility of the candidate",
        {"passed": ("admit_dataset",), "valid_failed": ()},
        required_context_fields=("candidate_dataset_manifest_hash",),
    ),
    "conditional_dataset_accepted": _contract(
        "conditional_dataset_accepted",
        "aggregate the three formal dataset evidence records",
        {"passed": ()},
        required_context_fields=("evaluator_checkpoint_hash", "dataset_split_manifest_hash", "checkpoint_hash"),
    ),
    "formal_evaluation_exposure": _contract(
        "formal_evaluation_exposure",
        "record one explicit official-test exposure in the evaluation lineage",
        {"passed": ("evaluate_official_test", "record_next_official_test_exposure")},
        required_context_fields=("evaluator_checkpoint_hash", "dataset_split_manifest_hash"),
        required_input_keys=("exposure_chain_id", "exposure_sequence", "test_manifest", "measurement_evidence"),
    ),
    # Compatibility output remains readable but can never authorize current
    # evaluation or promotion.
    "conditional_training_completion": _contract(
        "conditional_training_completion",
        "historical checkpoint metadata correction",
        {"passed": ("diversity_observability",)},
        formal=False,
    ),
}


def _context_value(context: EvidenceContextV2, name: str) -> object:
    return getattr(context, name)


def validate_current_gate_record(record: EvidenceRecordV2) -> CurrentGateContract:
    contract = CURRENT_GATE_CONTRACTS.get(record.evidence_kind)
    if contract is None:
        raise ValueError(f"current gate is not registered: {record.evidence_kind}")
    record.validate()
    if contract.formal and record.status == "failed":
        raise ValueError("current formal gate cannot use ambiguous failed status")
    if record.tier == "diagnostic" and contract.formal and record.permitted_next_actions:
        raise ValueError("diagnostic evidence cannot authorize a formal current gate")
    allowed = contract.permitted_by_status.get(record.status)
    if allowed is None:
        raise ValueError(f"status {record.status} is not valid for {record.evidence_kind}")
    if any(action not in allowed for action in record.permitted_next_actions):
        raise ValueError(f"permitted action is not valid for {record.evidence_kind}")
    for field in contract.required_context_fields:
        if _context_value(record.context, field) is None:
            raise ValueError(f"{record.evidence_kind} lacks context binding {field}")
    for key in contract.required_input_keys:
        if not isinstance(record.input_hashes.get(key), str):
            raise ValueError(f"{record.evidence_kind} lacks input binding {key}")
    return contract


def require_current_gate_action(
    record: EvidenceRecordV2,
    *,
    kind: str,
    action: str,
    expected_context: EvidenceContextV2,
) -> None:
    if record.evidence_kind != kind:
        raise ValueError(f"expected {kind} evidence")
    contract = validate_current_gate_record(record)
    record.context.assert_matches(expected_context)
    if action not in record.permitted_next_actions:
        raise ValueError(f"{kind} evidence does not authorize {action}")
    if not any(action in actions for actions in contract.permitted_by_status.values()):
        raise ValueError(f"{action} is not registered for {kind}")


def current_gate_outcome(kind: str, outcome: str) -> tuple[str, tuple[str, ...]]:
    contract = CURRENT_GATE_CONTRACTS[kind]
    actions = contract.permitted_by_status.get(outcome)
    if actions is None:
        raise ValueError(f"outcome {outcome} is not registered for {kind}")
    return outcome, actions
