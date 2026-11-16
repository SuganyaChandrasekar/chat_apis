from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

_SEMVER_PATTERN = (
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-((?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)
ManifestString = Annotated[str, Field(min_length=1, max_length=256)]


class RoutingManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    intents: list[ManifestString] = Field(min_length=1, max_length=100)
    keywords: list[ManifestString] = Field(max_length=100)


class CapabilityDataSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    required_inputs: list[ManifestString] = Field(max_length=100)
    output_type: str = Field(min_length=1, max_length=128)


class CapabilityManifest(BaseModel):
    model_config = ConfigDict(
        extra="forbid", str_strip_whitespace=True, populate_by_name=True
    )

    capability_id: str = Field(min_length=1, max_length=128)
    version: str = Field(pattern=_SEMVER_PATTERN, max_length=128)
    display_name: str = Field(min_length=1, max_length=256)
    description: str = Field(min_length=1, max_length=4000)
    entrypoint: str = Field(
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:"
        r"[A-Za-z_][A-Za-z0-9_]*$",
        max_length=512,
    )
    routing: RoutingManifest
    data_schema: CapabilityDataSchema = Field(alias="schema")


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=4000)
    expected_response: str = Field(min_length=1, max_length=4000)
    actual_response: str = Field(min_length=1, max_length=4000)


class CapabilityRegistrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest: CapabilityManifest
    evaluation_cases: list[EvaluationCase] = Field(min_length=1, max_length=200)


class CapabilityRegistrationResponse(BaseModel):
    capability_id: str
    version: str
    certification_status: str
    guardrails_passed: bool
    evaluation_score: float | None
    rollout_status: str
    feature_flag: str
    feature_flag_enabled: bool
    cohort: str
    general_traffic_percentage: int
    reasons: list[str]


class CapabilityPromotionResponse(BaseModel):
    capability_id: str
    version: str
    promoted: bool
    rollout_status: str
    candidate_score: float
    active_version: str | None
    reasons: list[str]


class CapabilityRouteResponse(BaseModel):
    capability_id: str
    version: str
    cohort: str
    feature_flag: str
    manifest: CapabilityManifest
