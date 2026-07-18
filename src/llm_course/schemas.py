from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

RelationType = Literal["builds_on", "improves", "contrasts_with", "used_by"]
PaperTier = Literal["core", "deep_dive", "frontier"]
PaperStatus = Literal["unread", "pass1", "pass2", "pass3", "reproduced"]


@dataclass(frozen=True)
class Relation:
    type: RelationType
    target: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Relation:
        return cls(type=data["type"], target=data["target"])


@dataclass(frozen=True)
class PaperRecord:
    id: str
    title: str
    year: int
    version_date: str
    source_type: str
    url: str
    code_url: str
    topics: tuple[str, ...]
    tier: PaperTier
    status: PaperStatus
    prerequisites: tuple[str, ...]
    claims: tuple[str, ...]
    evidence: tuple[str, ...]
    limitations: tuple[str, ...]
    reproduction: str
    relations: tuple[Relation, ...]
    as_of: str
    doi: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaperRecord:
        return cls(
            id=str(data["id"]),
            title=str(data["title"]),
            year=int(data["year"]),
            version_date=str(data["version_date"]),
            source_type=str(data["source_type"]),
            url=str(data["url"]),
            code_url=str(data.get("code_url", "")),
            topics=tuple(data["topics"]),
            tier=data["tier"],
            status=data["status"],
            prerequisites=tuple(data["prerequisites"]),
            claims=tuple(data["claims"]),
            evidence=tuple(data["evidence"]),
            limitations=tuple(data["limitations"]),
            reproduction=str(data["reproduction"]),
            relations=tuple(Relation.from_dict(item) for item in data["relations"]),
            as_of=str(data["as_of"]),
            doi=str(data.get("doi", "")),
        )


@dataclass(frozen=True)
class LessonManifest:
    week: int
    stage: str
    title: str
    objectives: tuple[str, ...]
    readings: tuple[str, ...]
    experiments: tuple[str, ...]
    exercise: str
    acceptance: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LessonManifest:
        return cls(
            week=int(data["week"]),
            stage=str(data["stage"]),
            title=str(data["title"]),
            objectives=tuple(data["objectives"]),
            readings=tuple(data["readings"]),
            experiments=tuple(data["experiments"]),
            exercise=str(data["exercise"]),
            acceptance=tuple(data["acceptance"]),
        )


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def merge(self, other: ValidationReport) -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

