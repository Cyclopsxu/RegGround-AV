"""从冻结输入确定性重放 JudgeInput v2 的 LLM 可见文本。

Layer 1 与 Layer 2 在固定 seed 下完全确定，因此正式运行时 LLM 实际读到的
prompt 片段可以在事后逐字重建，无需重跑 LLM。重建结果用运行日志里记录的
``feature_text_hash`` 逐条自校验：任何漂移都会立即抛错，而不是安静地把
"看起来差不多"的文本发给标注者或影子评估。

盲标包的证据区与影子评估的 hard filter 输入都只允许经由本模块取得。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from src.admission import admission_reason
from src.eval_set import load_eval_set
from src.layer1.facade import parse_scene
from src.layer1.models import RawScene, TrajectoryFeatures
from src.layer2.retriever import GraphRAGRetriever
from src.layer2.rule_graph import Layer2Settings
from src.layer3.context_builder import ContextBuilder
from src.layer3.models import JudgeContext, Layer3Settings
from src.trajectory_feature_text import feature_text_hash, render_feature_text

DEFAULT_RUN_LOG = Path("logs/eval_set_v2_full_v4_1.jsonl")
DEFAULT_INPUT = Path("data/layer1/raw_scene_trainval.json")
DEFAULT_EVAL_SET = Path("artifacts/eval_set_v1.json")
DEFAULT_RULE_GRAPH = Path("data/layer2/rule_graph.yaml")
DEFAULT_SEED = 42


class ReplayMismatchError(RuntimeError):
    """重建文本与运行日志记录的哈希不一致。"""


@dataclass(frozen=True)
class ReplayedScene:
    """一个 keyframe 的重放结果，含 LLM 逐字可见的上下文与候选特征。"""

    frame_token: str
    scene_id: str
    scenario_type: str
    context: JudgeContext
    features: list[TrajectoryFeatures]
    record: dict[str, Any]

    @property
    def trajectory_ids(self) -> list[str]:
        return list(self.context.trajectory_ids)

    def feature_by_id(self, trajectory_id: str) -> TrajectoryFeatures:
        index = self.trajectory_ids.index(trajectory_id)
        return self.features[index]

    def feature_text(self, trajectory_id: str) -> str:
        """HardFilter 提示词中该候选的逐字文本。"""
        return render_feature_text(self.feature_by_id(trajectory_id))

    def hard_filter_user_prompt(self, trajectory_id: str) -> str:
        """HardFilter 单候选调用时发给 LLM 的证据区。

        直接复用生产的 ``HardFilter.build_evidence_prompt``，不在此重写拼装
        逻辑；few-shot 与末尾固定指令段不属于证据，故不纳入。
        """
        from src.layer3.hard_filter import HardFilter

        index = self.trajectory_ids.index(trajectory_id)
        return HardFilter.build_evidence_prompt(
            self.context, [(index, self.features[index])]
        )

    def verdict_by_id(self, trajectory_id: str) -> dict[str, Any]:
        for item in self.record["final_label"]["verdicts"]:
            if item["trajectory_id"] == trajectory_id:
                return dict(item["veto"])
        raise KeyError(f"{self.frame_token} 缺少候选 {trajectory_id} 的判定")

    def benchmark_by_id(self, trajectory_id: str) -> dict[str, Any]:
        labels = self.record["layer1"]["benchmark_labels_debug_only"]["labels"]
        for item in labels:
            if item["anonymous_id"] == trajectory_id:
                return dict(item)
        raise KeyError(f"{self.frame_token} 缺少候选 {trajectory_id} 的 benchmark 标签")


def load_evaluated_records(run_log: Path = DEFAULT_RUN_LOG) -> dict[str, dict[str, Any]]:
    """读取正式运行日志中真正进入 Layer 3 的记录，按 frame_token 索引。"""
    records: dict[str, dict[str, Any]] = {}
    with run_log.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("type") != "record":
                continue
            data = item.get("data")
            if not isinstance(data, dict) or not data.get("final_label"):
                continue
            records[data["raw_scene"]["frame_token"]] = data
    if not records:
        raise ValueError(f"{run_log} 中没有可用记录")
    return records


def load_run_manifest(run_log: Path = DEFAULT_RUN_LOG) -> dict[str, Any]:
    """读取运行 manifest，用于核对重放所依赖的冻结输入。"""
    with run_log.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("type") == "manifest":
                return cast(dict[str, Any], item)
    raise ValueError(f"{run_log} 缺少 manifest 行")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_frozen_inputs_match(
    manifest: dict[str, Any],
    *,
    input_path: Path,
    eval_set_path: Path,
    rule_graph_path: Path = DEFAULT_RULE_GRAPH,
) -> None:
    """重放前确认冻结输入与产出该运行时逐字节一致。"""
    expected_input = manifest.get("input_sha256")
    actual_input = file_sha256(input_path)
    if expected_input != actual_input:
        raise ReplayMismatchError(
            f"冻结输入哈希不匹配：manifest={expected_input} 实际={actual_input}"
        )
    expected_eval = (manifest.get("eval_set") or {}).get("sha256")
    actual_eval = file_sha256(eval_set_path)
    if expected_eval != actual_eval:
        raise ReplayMismatchError(
            f"冻结评估集哈希不匹配：manifest={expected_eval} 实际={actual_eval}"
        )
    expected_graph = manifest.get("rule_graph_sha256")
    actual_graph = file_sha256(rule_graph_path)
    if expected_graph != actual_graph:
        raise ReplayMismatchError(
            f"冻结规则图谱哈希不匹配：manifest={expected_graph} 实际={actual_graph}"
        )


def _logged_feature_hashes(record: dict[str, Any]) -> dict[str, str]:
    for diagnostic in record["final_label"]["stage_diagnostics"]:
        if diagnostic["stage"] == "hard_filter":
            return dict(diagnostic.get("feature_text_hashes") or {})
    return {}


def replay_scenes(
    *,
    run_log: Path = DEFAULT_RUN_LOG,
    input_path: Path = DEFAULT_INPUT,
    eval_set_path: Path = DEFAULT_EVAL_SET,
    seed: int = DEFAULT_SEED,
    verify_hashes: bool = True,
) -> list[ReplayedScene]:
    """重放全部已评估 keyframe，并逐候选核对 LLM 可见文本哈希。"""
    manifest = load_run_manifest(run_log)
    assert_frozen_inputs_match(
        manifest,
        input_path=input_path,
        eval_set_path=eval_set_path,
        rule_graph_path=DEFAULT_RULE_GRAPH,
    )
    records = load_evaluated_records(run_log)

    eval_set = load_eval_set(eval_set_path, require_frozen=True)
    wanted = {entry.frame_token for entry in eval_set.entries} & set(records)

    with input_path.open(encoding="utf-8") as handle:
        raw_records = json.load(handle)
    by_token = {
        item["frame_token"]: item
        for item in raw_records
        if isinstance(item, dict) and item.get("frame_token") in wanted
    }
    missing = wanted - set(by_token)
    if missing:
        raise ReplayMismatchError(f"冻结输入缺少 {len(missing)} 个 frame_token")

    layer3_settings = Layer3Settings()
    builder = ContextBuilder(layer3_settings)
    retriever = GraphRAGRetriever(Layer2Settings())

    scenes: list[ReplayedScene] = []
    for frame_token in sorted(wanted):
        record = records[frame_token]
        raw_scene = RawScene.model_validate(by_token[frame_token])
        bundle = parse_scene(raw_scene, seed=seed)
        if admission_reason(bundle.scene_query) is not None:
            raise ReplayMismatchError(
                f"{frame_token} 在重放中被准入拒绝，但运行日志里有判定结果"
            )
        subgraph = retriever.retrieve(bundle.scene_query)
        context = builder.build(bundle.judge_input, subgraph)
        features = list(
            bundle.judge_input.trajectory_features[
                : layer3_settings.max_trajectories_per_judge
            ]
        )

        if verify_hashes:
            logged = _logged_feature_hashes(record)
            for trajectory_id, feature in zip(
                context.trajectory_ids, features, strict=True
            ):
                actual = feature_text_hash(feature)
                expected = logged.get(trajectory_id)
                if expected != actual:
                    raise ReplayMismatchError(
                        "重放文本与运行日志不一致："
                        f"frame={frame_token} traj={trajectory_id} "
                        f"日志={expected} 重放={actual}"
                    )
            logged_hard = [
                rule["node_id"] for rule in record["layer2"]["hard_rules"]
            ]
            if logged_hard != list(context.hard_rule_ids):
                raise ReplayMismatchError(
                    "重放硬规则集合与运行日志不一致："
                    f"frame={frame_token} 日志={logged_hard} 重放={context.hard_rule_ids}"
                )

        scenes.append(
            ReplayedScene(
                frame_token=frame_token,
                scene_id=record["raw_scene"]["scene_id"],
                scenario_type=record["layer1"]["context"]["scenario_type"],
                context=context,
                features=features,
                record=record,
            )
        )
    return scenes


__all__ = [
    "DEFAULT_EVAL_SET",
    "DEFAULT_INPUT",
    "DEFAULT_RUN_LOG",
    "DEFAULT_RULE_GRAPH",
    "DEFAULT_SEED",
    "ReplayMismatchError",
    "ReplayedScene",
    "assert_frozen_inputs_match",
    "file_sha256",
    "load_evaluated_records",
    "load_run_manifest",
    "replay_scenes",
]
