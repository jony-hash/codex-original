"""Explicit official-test exposure lineage for current formal evaluation."""
from __future__ import annotations

from ...evidence import EvidenceRecordV2, canonical_sha256
from ...config import V2Config
from .gates import require_current_gate_action
from .runtime import _context


EXPOSURE_CONTRACT = "formal_evaluation_exposure_chain_v1"


def exposure_chain_id(
    config: V2Config,
    *,
    primary_evaluator: str,
    secondary_evaluator: str,
    test_manifest: str,
    measurement_bundle: str,
) -> str:
    return canonical_sha256(
        {
            "contract": EXPOSURE_CONTRACT,
            "config": canonical_sha256(config.model_dump(mode="json")),
            "primary_evaluator": primary_evaluator,
            "secondary_evaluator": secondary_evaluator,
            "test_manifest": test_manifest,
            "measurement_bundle": measurement_bundle,
        }
    )


def require_exposure_predecessor(
    predecessor: EvidenceRecordV2,
    *,
    config: V2Config,
    primary_evaluator: str,
    secondary_evaluator: str,
    test_manifest: str,
    measurement_evidence: str,
    chain_id: str,
    split_hash: str,
) -> int:
    require_current_gate_action(
        predecessor,
        kind="formal_evaluation_exposure",
        action="record_next_official_test_exposure",
        expected_context=_context(
            config,
            evaluator_hash=primary_evaluator,
            split_hash=split_hash,
            checkpoint_hash=predecessor.context.checkpoint_hash,
        ),
    )
    if predecessor.input_hashes.get("exposure_chain_id") != chain_id:
        raise ValueError("official-test exposure chain mismatch")
    if predecessor.input_hashes.get("exposure_contract") != EXPOSURE_CONTRACT:
        raise ValueError("official-test exposure contract mismatch")
    if predecessor.input_hashes.get("primary_evaluator") != primary_evaluator:
        raise ValueError("official-test exposure primary evaluator mismatch")
    if predecessor.input_hashes.get("secondary_evaluator") != secondary_evaluator:
        raise ValueError("official-test exposure secondary evaluator mismatch")
    if predecessor.input_hashes.get("test_manifest") != test_manifest:
        raise ValueError("official-test exposure test manifest mismatch")
    sequence = predecessor.input_hashes.get("exposure_sequence")
    if not isinstance(sequence, str) or not sequence.isdigit():
        raise ValueError("official-test exposure sequence is malformed")
    if int(sequence) > 0 and predecessor.input_hashes.get("measurement_evidence") != measurement_evidence:
        raise ValueError("official-test exposure measurement mismatch")
    if int(sequence) > 0:
        previous = predecessor.input_hashes.get("previous_exposure")
        if not isinstance(previous, str) or len(previous) != 64:
            raise ValueError("official-test exposure predecessor binding is malformed")
    return int(sequence)
