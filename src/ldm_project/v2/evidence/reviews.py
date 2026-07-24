"""Strict, content-bound manual review declarations."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .io import file_sha256


class ManualReviewV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=2)
    review_kind: str
    reviewed_tensor_sha256: str
    reviewed_preview_sha256: str
    readable_labels: tuple[int, ...]
    reviewer: str = Field(min_length=1)

    @field_validator("reviewer")
    @classmethod
    def reviewer_is_completed(cls, value: str) -> str:
        if value == "REQUIRED_REVIEWER":
            raise ValueError("manual review template has not been completed")
        return value

    def validate_contract(self, *, kind: str, tensor_hash: str, preview_hash: str) -> None:
        if self.schema_version != 2 or self.review_kind != kind:
            raise ValueError("manual review kind does not match the requested gate")
        if self.reviewed_tensor_sha256 != tensor_hash:
            raise ValueError("manual review is not bound to the replayed raw-CMOS tensor")
        if self.reviewed_preview_sha256 != preview_hash:
            raise ValueError("manual review is not bound to the fixed-exposure preview")
        if any(label < 0 or label > 9 for label in self.readable_labels) or len(set(self.readable_labels)) != len(self.readable_labels):
            raise ValueError("manual review labels must be a unique subset of 0 through 9")


def load_manual_review(path: str | Path, *, kind: str, tensor_hash: str, preview_hash: str) -> tuple[ManualReviewV2, str]:
    review_path = Path(path)
    review = ManualReviewV2.model_validate_json(review_path.read_text(encoding="utf-8"))
    review.validate_contract(kind=kind, tensor_hash=tensor_hash, preview_hash=preview_hash)
    return review, file_sha256(review_path)


class ConditionalManualReviewV2(BaseModel):
    """Content-bound 16-samples-per-label review for conditional generation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=2)
    review_kind: str
    reviewed_tensor_sha256: str
    reviewed_preview_sha256: str
    samples_per_label: int
    readable_counts: tuple[int, ...]
    reviewer: str = Field(min_length=1)

    @field_validator("reviewer")
    @classmethod
    def completed_reviewer(cls, value: str) -> str:
        if value == "REQUIRED_REVIEWER":
            raise ValueError("manual review template has not been completed")
        return value

    def validate_contract(self, *, tensor_hash: str, preview_hash: str) -> None:
        if self.schema_version != 2 or self.review_kind != "conditional_generation":
            raise ValueError("conditional manual review kind does not match")
        if self.reviewed_tensor_sha256 != tensor_hash or self.reviewed_preview_sha256 != preview_hash:
            raise ValueError("conditional manual review is not bound to its raw-CMOS tensor and fixed-exposure preview")
        if self.samples_per_label != 16 or len(self.readable_counts) != 10:
            raise ValueError("conditional manual review requires sixteen samples and ten readable counts")
        if any(count < 0 or count > self.samples_per_label for count in self.readable_counts):
            raise ValueError("conditional readable counts must be within [0, 16]")


def load_conditional_manual_review(path: str | Path, *, tensor_hash: str, preview_hash: str) -> tuple[ConditionalManualReviewV2, str]:
    review_path = Path(path)
    review = ConditionalManualReviewV2.model_validate_json(review_path.read_text(encoding="utf-8"))
    review.validate_contract(tensor_hash=tensor_hash, preview_hash=preview_hash)
    return review, file_sha256(review_path)
