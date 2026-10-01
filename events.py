"""Load independently produced layer-event annotations for semantic evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.fuzzy_dynamics.config import REASONING_MODES


def _read_event_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    elif path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        records = value if isinstance(value, list) else [value]
    else:
        raise ValueError("Semantic events must be a .json or .jsonl file.")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Every semantic-event record must be a JSON object.")
    return records


def load_semantic_event_definitions(
    path: str | Path,
    metadata: list[dict[str, Any]],
    num_layers: int,
    id_field: str = "uid",
    allow_missing: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Align binary transition-layer events to cache metadata by stable ID.

    Expected JSON/JSONL records have the form::

        {"uid": "q1", "events": {"hop_transition": [0, null, 1, ...]},
         "provenance": {"source": "human-v1", "independent_of_training": true}}

    Event arrays have length ``num_layers`` and align with transitions
    ``s_l -> s_(l+1)``. ``null`` marks an unlabelled layer.
    """
    if num_layers <= 0:
        raise ValueError("num_layers must be positive.")
    records = _read_event_records(Path(path))
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        if id_field not in record:
            raise ValueError(f"Semantic-event record is missing id field '{id_field}'.")
        identifier = str(record[id_field])
        if identifier in by_id:
            raise ValueError(f"Duplicate semantic-event ID: {identifier}")
        by_id[identifier] = record

    labels = {
        mode: torch.zeros((len(metadata), num_layers), dtype=torch.bool)
        for mode in REASONING_MODES
    }
    valid = {
        mode: torch.zeros((len(metadata), num_layers), dtype=torch.bool)
        for mode in REASONING_MODES
    }
    independent = {mode: True for mode in REASONING_MODES}
    sources = {mode: set() for mode in REASONING_MODES}
    missing_ids: list[str] = []
    matched = 0
    selected_ids: set[str] = set()

    for query_index, item in enumerate(metadata):
        if id_field not in item:
            raise ValueError(f"Activation metadata is missing id field '{id_field}'.")
        identifier = str(item[id_field])
        if identifier in selected_ids:
            raise ValueError(f"Duplicate activation-metadata ID: {identifier}")
        selected_ids.add(identifier)
        record = by_id.get(identifier)
        if record is None:
            missing_ids.append(identifier)
            continue
        matched += 1
        events = record.get("events")
        if not isinstance(events, dict):
            raise ValueError(f"Semantic-event record {identifier} has no 'events' object.")
        provenance = record.get("provenance", {})
        if not isinstance(provenance, dict):
            provenance = {}
        record_independent = bool(provenance.get("independent_of_training", False))
        source = str(provenance.get("source", "external_unspecified"))

        for mode in REASONING_MODES:
            values = events.get(mode)
            if values is None:
                continue
            if not isinstance(values, list) or len(values) != num_layers:
                raise ValueError(
                    f"Event '{mode}' for {identifier} must contain exactly "
                    f"{num_layers} transition-layer values."
                )
            independent[mode] &= record_independent
            sources[mode].add(source)
            for layer, value in enumerate(values):
                if value is None:
                    continue
                if value not in (0, 1, False, True):
                    raise ValueError(
                        f"Event '{mode}' for {identifier}, layer {layer} must be 0, 1, or null."
                    )
                labels[mode][query_index, layer] = bool(value)
                valid[mode][query_index, layer] = True

    if missing_ids and not allow_missing:
        preview = ", ".join(missing_ids[:5])
        raise ValueError(
            f"No semantic-event record for {len(missing_ids)} selected queries "
            f"({preview}). Use --allow-missing-events to mask them."
        )

    definitions = {
        mode: {
            "labels": labels[mode],
            "valid_mask": valid[mode],
            "source": ",".join(sorted(sources[mode])) or "external_unlabelled",
            "independent": independent[mode] and bool(sources[mode]),
        }
        for mode in REASONING_MODES
    }
    manifest = {
        "path": str(Path(path).resolve()),
        "id_field": id_field,
        "layer_axis": "transition_s_l_to_s_l_plus_1",
        "num_layers": num_layers,
        "records_in_file": len(records),
        "selected_queries": len(metadata),
        "matched_queries": matched,
        "missing_queries": len(missing_ids),
        "allow_missing": allow_missing,
    }
    return definitions, manifest
