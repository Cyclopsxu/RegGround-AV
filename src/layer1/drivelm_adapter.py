"""DriveLM QA schema adapters shared by dataset build and trainval spike paths."""

from __future__ import annotations

from typing import Any


def flatten_drivelm_qa(drivelm_frame: dict[str, Any] | None) -> list[dict[str, str]]:
    """Flatten DriveLM's category-indexed QA payload into Layer 1 qa_pairs."""
    if not drivelm_frame:
        return []
    qa_by_category = drivelm_frame.get("QA", {})
    if not isinstance(qa_by_category, dict):
        return []

    qa_pairs: list[dict[str, str]] = []
    for category in ("perception", "prediction", "planning", "behavior"):
        items = qa_by_category.get(category, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            question = item.get("Q", item.get("question", ""))
            answer = item.get("A", item.get("answer", ""))
            if not isinstance(question, str):
                question = ""
            if not isinstance(answer, str):
                answer = ""
            if question or answer:
                qa_pairs.append({
                    "category": category,
                    "question": question,
                    "answer": answer,
                })
    return qa_pairs


def normalized_traffic_status(obj_info: dict[str, Any]) -> str | None:
    """Normalize traffic-light phase from status or visual description."""
    raw_status = obj_info.get("Status", obj_info.get("status"))
    if isinstance(raw_status, str) and raw_status:
        status_text = raw_status.lower()
    else:
        visual = obj_info.get("Visual_description", obj_info.get("visual_description", ""))
        status_text = visual.lower() if isinstance(visual, str) else ""

    if "red" in status_text:
        return "red"
    if "yellow" in status_text or "amber" in status_text:
        return "yellow"
    if "green" in status_text:
        return "green"
    return None


def normalize_drivelm_key_object_infos(
    drivelm_frame: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Normalize v1.0/v1.1 DriveLM key-object fields for Layer 1 consumers."""
    if not drivelm_frame:
        return {}
    raw_infos = drivelm_frame.get("key_object_infos", {})
    if not isinstance(raw_infos, dict):
        return {}

    normalized: dict[str, dict[str, Any]] = {}
    for object_id, obj_info in raw_infos.items():
        if not isinstance(object_id, str) or not isinstance(obj_info, dict):
            continue
        item = dict(obj_info)
        category = obj_info.get("Category", obj_info.get("category", ""))
        visual = obj_info.get("Visual_description", obj_info.get("visual_description", ""))
        category_text = category.lower() if isinstance(category, str) else ""
        visual_text = visual.lower() if isinstance(visual, str) else ""

        if "traffic" in category_text and "light" in visual_text:
            item["category"] = "traffic_light"
            status = normalized_traffic_status(obj_info)
            if status is not None:
                item["status"] = status
                item["Status"] = status
        elif isinstance(category, str):
            item["category"] = category

        if "Visual_description" in item and "visual_description" not in item:
            item["visual_description"] = item["Visual_description"]
        normalized[object_id] = item
    return normalized


def normalize_drivelm_frame(drivelm_frame: dict[str, Any]) -> dict[str, Any]:
    """Preserve raw DriveLM fields while adding Layer 1's normalized views."""
    normalized = dict(drivelm_frame)
    existing_qa_pairs = drivelm_frame.get("qa_pairs")
    normalized["qa_pairs"] = (
        existing_qa_pairs
        if isinstance(existing_qa_pairs, list)
        else flatten_drivelm_qa(drivelm_frame)
    )
    normalized["key_object_infos"] = normalize_drivelm_key_object_infos(drivelm_frame)
    return normalized
