"""Pydantic data models for the ICTV classifier API."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    fasta: str = Field(..., description="FASTA-format sequence(s)")
    sequence_type: str = Field("auto", description="'nt', 'aa', or 'auto'")
    max_steps: int = Field(20, description="Max agent reasoning steps")
    family_hint: str = Field("", description="Optional: specify virus family to skip BLAST step")


class TaxonomyResult(BaseModel):
    realm: Optional[str] = None
    kingdom: Optional[str] = None
    phylum: Optional[str] = None
    class_: Optional[str] = Field(None, alias="class")
    order: Optional[str] = None
    family: Optional[str] = None
    subfamily: Optional[str] = None
    genus: Optional[str] = None
    subgenus: Optional[str] = None
    species: Optional[str] = None
    genome: Optional[str] = None

    class Config:
        populate_by_name = True


class Evidence(BaseModel):
    method: str
    region: str
    value: Any
    threshold: Any
    conclusion: str


class ClassifyResult(BaseModel):
    query_id: str
    taxonomy: TaxonomyResult
    confidence: str  # "High" | "Medium" | "Low" | "Unknown"
    novel_species: bool = False
    evidence: list[Evidence] = []
    reasoning: str = ""
    top_blast_hits: list[dict] = []
    criteria_used: Optional[dict] = None
    extracted_regions: list[dict] = []  # [{source, region, type, sequence}]
    token_usage: dict = {}  # {input_tokens, output_tokens, cache_read, cache_created, api_calls}


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    error = "error"


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    result: Optional[ClassifyResult] = None
    error: Optional[str] = None
    steps: list[str] = []  # reasoning step log
