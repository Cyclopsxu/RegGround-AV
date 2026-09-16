"""冻结正式评估集的读写与校验。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EVAL_SET_VERSION: Literal["eval_set_v1"] = "eval_set_v1"
MINI_EXCLUSION_REASON = (
    "10 个 nuScenes mini 场景是开发集；谓词、prompt、阈值和人工审计均在其上迭代。"
)
SELECTION_RULE = (
    "先排除 mini 场景；对每个 (scene_token, scenario_type) 准入单元，选择 horizon 内 "
    "ego 位移最大的帧。平手时选择时间戳更早的帧，再按 frame_token 字典序打破平手。"
    "选择过程没有随机成分；seed=42 仅保留给未来显式子抽样。"
)
THIN_LAYER_DECISIONS = {
    "emergency": "定性案例研究：照跑但不进入聚合指标；论文选择 1–2 例展示完整链路。",
    "yellow_light": "独立披露层：逐场景报告结果，明确准入后 n=12，并预声明不作显著性主张。",
    "singapore": "永久排除：左行与右行对应不同规则图，不作为 fallback 数据源。",
}
_CLASSIFICATION_SOURCE_FILES = (
    "src/admission.py",
    "src/condition_mapping.py",
    "src/layer1/scene_fact_extractor.py",
    "src/layer1/scene_text_extractor.py",
)
_PROJECT_ROOT = Path(__file__).parent.parent


class EvalSetEntry(BaseModel):
    """正式评估集内一个冻结的场景 × scenario 单元。"""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    frame_token: str
    scene_token: str
    scenario_type: str
    horizon_displacement_m: float = Field(ge=0.0)
    timestamp: int = Field(ge=0)


class EvalSet(BaseModel):
    """正式评估集五件套及其冻结状态。"""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    version: Literal["eval_set_v1"] = EVAL_SET_VERSION
    state: Literal["draft", "frozen"]
    entries: list[EvalSetEntry]
    selection_rule: str = SELECTION_RULE
    mini_scene_tokens: list[str]
    mini_exclusion_reason: str = MINI_EXCLUSION_REASON
    thin_layer_decisions: dict[str, str] = Field(
        default_factory=lambda: dict(THIN_LAYER_DECISIONS),
    )
    generation_script: str
    generation_commit: str
    spike_report: str
    input_source: str
    input_sha256: str
    input_provenance: str
    classification_source_files: list[str]
    classification_code_sha256: str
    determinism_verified: bool = False


def load_mini_scene_tokens(path: Path) -> list[str]:
    """读取 nuScenes mini 的完整 scene token 清单。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} top-level JSON must be a list")
    tokens = sorted(
        item["token"]
        for item in data
        if isinstance(item, dict) and isinstance(item.get("token"), str) and item["token"]
    )
    if len(tokens) != 10 or len(set(tokens)) != 10:
        raise ValueError(f"{path} must contain exactly 10 unique mini scene tokens")
    return tokens


def load_eval_set(path: Path, *, require_frozen: bool = False) -> EvalSet:
    """读取评估集；正式运行时拒绝未冻结草稿。"""
    eval_set = EvalSet.model_validate_json(path.read_text(encoding="utf-8"))
    _validate_entries(eval_set)
    if require_frozen and eval_set.state != "frozen":
        raise ValueError(f"{path} is a draft; formal evaluation requires a frozen eval set")
    return eval_set


def write_eval_set(path: Path, eval_set: EvalSet) -> None:
    """以稳定 JSON 格式写出评估集，支持逐字节可复现性检查。"""
    _validate_entries(eval_set)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(eval_set_json_bytes(eval_set))


def eval_set_json_bytes(eval_set: EvalSet) -> bytes:
    """返回稳定序列化结果，供双跑逐字节验证复用。"""
    _validate_entries(eval_set)
    return (
        json.dumps(eval_set.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")


def classification_code_sha256() -> str:
    """计算决定场景分类与准入的代码指纹。"""
    digest = hashlib.sha256()
    for relative_path in _CLASSIFICATION_SOURCE_FILES:
        path = _PROJECT_ROOT / relative_path
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def classification_source_files() -> list[str]:
    """返回分类指纹覆盖范围，供冻结文件与运行时显式核对。"""
    return list(_CLASSIFICATION_SOURCE_FILES)


def assert_classification_code_matches(eval_set: EvalSet) -> None:
    """拒绝分类/准入代码与冻结集合生成时不一致的正式运行。"""
    if eval_set.classification_source_files != classification_source_files():
        raise ValueError(
            "classification source file list does not match the frozen eval set",
        )
    if eval_set.classification_code_sha256 != classification_code_sha256():
        raise ValueError(
            "classification/admission code does not match the frozen eval set fingerprint",
        )


def file_sha256(path: Path) -> str:
    """计算输入或产物的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_input_file_matches(eval_set: EvalSet, input_path: Path) -> None:
    """拒绝与冻结集合所钉输入数据不一致的正式运行。"""
    if eval_set.input_sha256 != file_sha256(input_path):
        raise ValueError(
            "prepared input SHA-256 does not match the frozen eval set",
        )


def _validate_entries(eval_set: EvalSet) -> None:
    pairs = {(entry.scene_token, entry.scenario_type) for entry in eval_set.entries}
    frame_tokens = {entry.frame_token for entry in eval_set.entries}
    if len(pairs) != len(eval_set.entries):
        raise ValueError("eval set contains duplicate (scene_token, scenario_type) entries")
    if len(frame_tokens) != len(eval_set.entries):
        raise ValueError("eval set contains duplicate frame_token entries")
    mini_tokens = set(eval_set.mini_scene_tokens)
    if len(mini_tokens) != 10 or len(mini_tokens) != len(eval_set.mini_scene_tokens):
        raise ValueError("eval set must record exactly 10 unique mini scene tokens")
    if any(entry.scene_token in mini_tokens for entry in eval_set.entries):
        raise ValueError("eval set contains a mini scene")
    if eval_set.state == "frozen" and not eval_set.determinism_verified:
        raise ValueError("frozen eval set requires a successful byte-identical double run")
