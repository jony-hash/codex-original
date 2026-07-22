"""Strict, content-bound manual review declarations."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .io import file_sha256


class ManualReviewV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=2)
    review_kind: str
    reviewed_tensor_sha256: str
    readable_labels: tuple[int, ...]
    reviewer: str = Field(min_length=1)

    def validate_contract(self, *, kind: str, tensor_hash: str) -> None:
        if self.schema_version != 2 or self.review_kind != kind:
            raise ValueError("manual review kind does not match the requested gate")
        if self.reviewed_tensor_sha256 != tensor_hash:
            raise ValueError("manual review is not bound to the replayed raw-CMOS tensor")
        if list(self.readable_labels) != list(range(10)):
            raise ValueError("manual review must contain exactly readable labels 0 through 9")


def load_manual_review(path: str | Path, *, kind: str, tensor_hash: str) -> tuple[ManualReviewV2, str]:
    review_path = Path(path)
    review = ManualReviewV2.model_validate_json(review_path.read_text(encoding="utf-8"))
    review.validate_contract(kind=kind, tensor_hash=tensor_hash)
    return review, file_sha256(review_path)
