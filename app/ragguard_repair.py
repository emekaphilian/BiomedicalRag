"""Narrow, request-scoped repair contract for RAGGuard."""
from __future__ import annotations

import hmac
import json
import logging
from dataclasses import dataclass
from time import monotonic
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.ragguard_observation import get_server_setting


logger = logging.getLogger("biomedicalrag.ragguard_repair")
SUPPORTED_REPAIRS = ("HYBRID_RETRIEVAL", "RERANK")
REPAIR_CONTEXT_TTL_SECONDS = 300
MAX_REPAIR_CONTEXTS_PER_SERVICE = 256


class RepairRetrievedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=200)
    score: float


class RepairRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["v1"] = "v1"
    request_id: str = Field(min_length=1, max_length=200)
    application_id: str = Field(min_length=1, max_length=100)
    environment: str = Field(min_length=1, max_length=80)
    # RAGGuard sends the redacted observation query. Execution uses the
    # short-lived local request context, never this field.
    query: str = Field(min_length=1, max_length=20_000)
    failure_type: str = Field(min_length=1, max_length=80)
    strategy: Literal["DEDUPLICATE", "HYBRID_RETRIEVAL", "RERANK"]
    retrieved_chunks: list[RepairRetrievedChunk] = Field(max_length=500)


class RepairResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["v1"] = "v1"
    request_id: str
    strategy: Literal["DEDUPLICATE", "HYBRID_RETRIEVAL", "RERANK"]
    status: Literal["promoted", "rejected", "rolled_back"]
    validated: bool
    context_verified: bool
    improvement: float | None = Field(default=None, allow_inf_nan=False)


class RepairCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["v1"] = "v1"
    supported_repairs: list[Literal["HYBRID_RETRIEVAL", "RERANK"]]


@dataclass
class PendingRepair:
    created_at: float
    question: str
    mode: str
    top_k: int
    chat_history: list | None
    retrieval_query: str | None
    original_details: dict
    result: RepairResult | None = None
    strategy: str | None = None
    attempted: bool = False


def authorize_repair(authorization: str | None) -> bool:
    expected = get_server_setting("RAGGUARD_REPAIR_TOKEN")
    if not expected:
        return False
    scheme, _, supplied = (authorization or "").partition(" ")
    return bool(
        scheme.lower() == "bearer"
        and supplied
        and hmac.compare_digest(supplied, expected)
    )


def audit_repair(
    *,
    request_id: str,
    strategy: str,
    authorized: bool,
    outcome: str,
    validated: bool | None,
    context_verified: bool | None,
) -> None:
    logger.info(
        "RAGGuard repair audit %s",
        json.dumps(
            {
                "request_id": request_id,
                "strategy": strategy,
                "authorized": authorized,
                "outcome": outcome,
                "validated": validated,
                "context_verified": context_verified,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
