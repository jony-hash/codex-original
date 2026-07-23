"""Content-addressed LightGen reference claims, kept separate from formal v2 optics."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping

from .evidence import canonical_sha256


SourceStatus = Literal["paper_reported", "user_locked", "project_approximation", "unknown"]


@dataclass(frozen=True)
class LightGenReferenceFieldV2:
    value: object | None
    status: SourceStatus
    source: str | None

    def validate(self) -> None:
        if self.status == "unknown" and self.value is not None:
            raise ValueError("unknown LightGen reference fields must not invent a value")
        if self.status != "unknown" and not self.source:
            raise ValueError("known LightGen reference fields require a source")


@dataclass(frozen=True)
class LightGenReferenceContractV2:
    schema_version: int
    paper_doi: str
    pubmed_url: str
    zenodo_doi: str
    zenodo_record_url: str
    fields: Mapping[str, LightGenReferenceFieldV2]

    @property
    def contract_hash(self) -> str:
        return canonical_sha256(self.to_dict())

    @property
    def architecture_proxy_only(self) -> bool:
        return any(field.status == "unknown" for field in self.fields.values())

    def validate(self) -> None:
        if self.schema_version != 2 or self.paper_doi != "10.1126/science.adv7434":
            raise ValueError("unsupported LightGen reference contract")
        if not self.zenodo_doi.startswith("10.5281/zenodo."):
            raise ValueError("LightGen contract requires a Zenodo DOI")
        for field in self.fields.values():
            field.validate()

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["contract_hash"] = self.contract_hash
        return values

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "LightGenReferenceContractV2":
        fields = {name: LightGenReferenceFieldV2(**payload) for name, payload in dict(value["fields"]).items()}
        contract = cls(int(value["schema_version"]), str(value["paper_doi"]), str(value["pubmed_url"]), str(value["zenodo_doi"]), str(value["zenodo_record_url"]), fields)
        contract.validate()
        supplied = value.get("contract_hash")
        if supplied is not None and supplied != contract.contract_hash:
            raise ValueError("LightGen reference contract hash changed")
        return contract
