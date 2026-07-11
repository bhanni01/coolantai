"""Heuristic prompt-injection detection with logging, never blocking.

These patterns have false positives (a legitimate PDF can discuss "system
prompts"), so matches are logged for visibility and the pipeline continues.
Configure the "coolant_copilot.injection_audit" logger to route the warnings.
"""

import logging
from collections.abc import Iterable

from langchain_core.documents import Document

from coolant_copilot.state import TargetSpec

logger = logging.getLogger("coolant_copilot.injection_audit")

SUSPICIOUS_PATTERNS: tuple[str, ...] = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "ignore the above",
    "disregard the above",
    "disregard previous instructions",
    "system prompt",
    "you are now",
    "new instructions:",
    "do not follow",
)


def scan_text(text: str, source: str) -> list[str]:
    """Log every suspicious pattern found in text; return the matches."""
    lowered = text.lower()
    hits = [p for p in SUSPICIOUS_PATTERNS if p in lowered]
    for hit in hits:
        logger.warning("possible prompt injection in %s: matched %r", source, hit)
    return hits


def scan_target_spec(spec: TargetSpec) -> list[str]:
    """Scan every free-text field of an incoming TargetSpec."""
    hits = scan_text(spec.name, "target_spec.name")
    hits += scan_text(spec.description, "target_spec.description")
    for i, substance in enumerate(spec.excluded_substances):
        hits += scan_text(substance, f"target_spec.excluded_substances[{i}]")
    return hits


def scan_documents(docs: Iterable[Document]) -> list[str]:
    """Scan document content (newly ingested chunks or extraction input)."""
    hits: list[str] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        hits += scan_text(doc.page_content, f"document '{source}'")
    return hits
