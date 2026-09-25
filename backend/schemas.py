from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Comment(StrictModel):
    id: str
    text: str
    timestamp: str | None = None


class AnalyzeRequest(StrictModel):
    text: str = Field(max_length=10_000)
    comments: list[Comment] = Field(default_factory=list, max_length=30)

    image: str | None = None
    reference_image: str | None = None

    locale: str = "zh-CN"

    @field_validator("image", "reference_image")
    @classmethod
    def validate_image(cls, value):
        if value is None:
            return value

        allowed_prefixes = (
            "data:image/png;base64,",
            "data:image/jpeg;base64,",
            "data:image/webp;base64,",
            "upload_",
        )

        if not value.startswith(allowed_prefixes):
            raise ValueError(
                "Image must be a supported data URL or controlled upload ID"
            )

        return value


class Authorship(StrictModel):
    status: Literal[
        "not_yet_analyzed",
        "insufficient_evidence",
        "analyzed",
    ]

    risk_score: float | None = None
    probability_any_ai: float | None = None
    calibration_version: str | None = None


class Versions(StrictModel):
    agent: str


class AnalyzeResponse(StrictModel):
    report_id: str

    authorship: Authorship

    findings: list = Field(default_factory=list)

    decision: Literal[
        "request_more_evidence",
        "no_action",
        "review",
    ]

    recommended_actions: list[str] = Field(default_factory=list)

    trace: list = Field(default_factory=list)

    review_history: list = Field(default_factory=list)

    versions: Versions