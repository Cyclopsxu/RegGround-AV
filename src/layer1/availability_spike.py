"""trainval 数据可得性与正式准入口径预演。"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.admission import MIN_EGO_DISPLACEMENT_M, admission_reason
from src.eval_set import (
    THIN_LAYER_DECISIONS,
    EvalSet,
    EvalSetEntry,
    classification_code_sha256,
    classification_source_files,
    eval_set_json_bytes,
    file_sha256,
    load_mini_scene_tokens,
    write_eval_set,
)
from src.layer1.dataset_loader import DriveLMDatasetLoader
from src.layer1.models import RawScene, SceneQuery
from src.layer1.scene_fact_extractor import SceneFactExtractor
from src.layer1.scene_text_extractor import SceneTextExtractor


class CountDistribution(BaseModel):
    """每 keyframe 计数分布。"""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    minimum: int = 0
    median: float = 0.0
    p95: int = 0
    maximum: int = 0


class AvailabilityStats(BaseModel):
    """按生产准入逻辑计算的数据可得性统计。"""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    boston_drivelm_keyframes: int = 0
    total_keyframes: int = 0
    unique_scenes: int = 0
    location_keyframes: dict[str, int] = Field(default_factory=dict)
    scenario_counts: dict[str, int] = Field(default_factory=dict)
    scenario_unique_scene_counts: dict[str, int] = Field(default_factory=dict)
    evaluable_scenario_counts: dict[str, int] = Field(default_factory=dict)
    admitted_scenario_unique_scene_counts: dict[str, int] = Field(default_factory=dict)
    admitted_pair_count: int = 0
    mini_admitted_pair_count: int = 0
    traffic_status_sources: dict[str, int] = Field(default_factory=dict)
    admission_reasons: dict[str, int] = Field(default_factory=dict)
    admitted_keyframes: int = 0
    admitted_unique_scenes: int = 0
    admitted_keyframes_per_scene: CountDistribution = Field(default_factory=CountDistribution)
    selected_eval_entries: list[EvalSetEntry] = Field(default_factory=list)
    stationary_keyframes: int = 0
    scenes_with_stationary_keyframes: int = 0
    stationary_admitted_keyframes: int = 0
    stationary_displacement_threshold_m: float = MIN_EGO_DISPLACEMENT_M
    qa_pair_counts: CountDistribution = Field(default_factory=CountDistribution)
    key_object_counts: CountDistribution = Field(default_factory=CountDistribution)
    stop_line_geometry_keyframes: int = 0
    ped_crossing_geometry_keyframes: int = 0
    skipped_keyframes: int = 0
    notes: list[str] = Field(default_factory=list)


class _PreparedDatasetLoader:
    """为已拼接 RawScene 输入提供与 DriveLM loader 相同的最小读取接口。"""

    def __init__(self, input_path: Path) -> None:
        data = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{input_path} top-level JSON must be a list")
        self._records: dict[tuple[str, str], dict[str, Any]] = {}
        self._location_counts: Counter[str] = Counter()
        for record in data:
            if not isinstance(record, dict):
                continue
            scene_token = record.get("scene_id")
            frame_token = record.get("frame_token")
            if not isinstance(scene_token, str) or not isinstance(frame_token, str):
                continue
            key = (scene_token, frame_token)
            if key in self._records:
                raise ValueError(f"duplicate prepared keyframe: {scene_token}/{frame_token}")
            self._records[key] = record
            location = record.get("location")
            self._location_counts[str(location) if location else "unknown"] += 1

    def list_keyframes(self) -> list[tuple[str, str]]:
        """列出已拼接输入中的场景与 keyframe。"""
        return sorted(self._records)

    def load_keyframe(self, scene_token: str, frame_token: str) -> RawScene:
        """恢复单条 RawScene，沿用生产模型校验。"""
        return RawScene.model_validate(self._records[(scene_token, frame_token)])

    def drivelm_keyframe_counts_by_location(self) -> dict[str, int]:
        """按已拼接输入中的 location 统计 keyframe。"""
        return dict(sorted(self._location_counts.items()))


def collect_availability_stats(
    nuscenes_root: Path,
    drivelm_qa_path: Path,
    *,
    stationary_displacement_threshold_m: float = MIN_EGO_DISPLACEMENT_M,
    mini_scene_tokens: set[str] | None = None,
) -> AvailabilityStats:
    """只运行生产事实提取、场景分类和准入逻辑，不调用 LLM。"""
    loader = DriveLMDatasetLoader(nuscenes_root, drivelm_qa_path)
    return _collect_availability_stats(
        loader,
        stationary_displacement_threshold_m=stationary_displacement_threshold_m,
        mini_scene_tokens=mini_scene_tokens,
    )


def collect_availability_stats_from_prepared_input(
    input_path: Path,
    *,
    stationary_displacement_threshold_m: float = MIN_EGO_DISPLACEMENT_M,
    mini_scene_tokens: set[str] | None = None,
) -> AvailabilityStats:
    """从正式 runner 使用的 RawScene JSON 重跑同一套 spike 口径。"""
    return _collect_availability_stats(
        _PreparedDatasetLoader(input_path),
        stationary_displacement_threshold_m=stationary_displacement_threshold_m,
        mini_scene_tokens=mini_scene_tokens,
    )


def _collect_availability_stats(
    loader: Any,
    *,
    stationary_displacement_threshold_m: float,
    mini_scene_tokens: set[str] | None,
) -> AvailabilityStats:
    """基于统一 loader 接口统计可得性，避免元数据与准备输入口径漂移。"""
    fact_extractor = SceneFactExtractor()
    text_extractor = SceneTextExtractor()
    scenarios: Counter[str] = Counter()
    evaluable_scenarios: Counter[str] = Counter()
    status_sources: Counter[str] = Counter()
    admission_reasons: Counter[str] = Counter()
    qa_counts: list[int] = []
    key_object_counts: list[int] = []
    admitted = 0
    stationary = 0
    stationary_admitted = 0
    stop_line = 0
    ped_crossing = 0
    skipped = 0
    seen_scenes: set[str] = set()
    admitted_scenes: set[str] = set()
    stationary_scenes: set[str] = set()
    scenario_scenes: dict[str, set[str]] = {}
    admitted_scenario_scenes: dict[str, set[str]] = {}
    admitted_by_scene: Counter[str] = Counter()
    admitted_frames: list[EvalSetEntry] = []

    for scene_token, frame_token in loader.list_keyframes():
        try:
            raw_scene = loader.load_keyframe(scene_token, frame_token)
            facts = fact_extractor.extract(raw_scene)
            scenario_type = text_extractor.infer_scenario_type(raw_scene, facts)
            description = text_extractor.extract(raw_scene, facts)
            query = SceneQuery(
                scene_id=raw_scene.scene_id,
                frame_token=raw_scene.frame_token,
                scenario_type=scenario_type,
                keywords=description.keywords,
                scene_facts=facts,
            )
        except Exception:
            skipped += 1
            continue

        scenario = scenario_type.value
        seen_scenes.add(scene_token)
        scenarios[scenario] += 1
        scenario_scenes.setdefault(scenario, set()).add(scene_token)
        status_sources[facts.traffic_light_status_source] += 1
        reason = admission_reason(query)
        if reason is None:
            admitted += 1
            admitted_scenes.add(scene_token)
            admitted_scenario_scenes.setdefault(scenario, set()).add(scene_token)
            admitted_by_scene[scene_token] += 1
            admitted_frames.append(EvalSetEntry(
                frame_token=frame_token,
                scene_token=scene_token,
                scenario_type=scenario,
                horizon_displacement_m=_ego_displacement_m(raw_scene.ego_poses),
                timestamp=_frame_timestamp(raw_scene),
            ))
        else:
            admission_reasons[reason] += 1

        displacement = _ego_displacement_m(raw_scene.ego_poses)
        is_stationary = displacement < stationary_displacement_threshold_m
        stationary += int(is_stationary)
        if is_stationary:
            stationary_scenes.add(scene_token)
        stationary_admitted += int(is_stationary and reason is None)
        if reason is None and not is_stationary:
            evaluable_scenarios[scenario] += 1

        qa_pairs = raw_scene.raw_json.get("qa_pairs", [])
        key_objects = raw_scene.raw_json.get("key_object_infos", {})
        qa_counts.append(len(qa_pairs) if isinstance(qa_pairs, list) else 0)
        key_object_counts.append(len(key_objects) if isinstance(key_objects, dict) else 0)
        stop_line += int(bool(raw_scene.map_records.get("stop_line")))
        ped_crossing += int(bool(raw_scene.map_records.get("ped_crossing")))

    notes = [
        f"{scenario} 非静止可评估样本 < 20；按冻结的薄层处置执行。"
        for scenario, count in sorted(evaluable_scenarios.items())
        if scenario != "unknown" and count < 20
    ]
    mini_scene_tokens = mini_scene_tokens or set()
    admitted_pairs = {(frame.scene_token, frame.scenario_type) for frame in admitted_frames}
    mini_admitted_pairs = {
        pair for pair in admitted_pairs if pair[0] in mini_scene_tokens
    }
    selected_eval_entries = select_representative_frames(
        [frame for frame in admitted_frames if frame.scene_token not in mini_scene_tokens],
    )
    if len(selected_eval_entries) != len(admitted_pairs) - len(mini_admitted_pairs):
        raise ValueError("selected eval set length does not match admitted pair count")
    return AvailabilityStats(
        boston_drivelm_keyframes=sum(scenarios.values()),
        total_keyframes=sum(scenarios.values()),
        unique_scenes=len(seen_scenes),
        location_keyframes=loader.drivelm_keyframe_counts_by_location(),
        scenario_counts=dict(sorted(scenarios.items())),
        scenario_unique_scene_counts={
            scenario: len(scene_tokens)
            for scenario, scene_tokens in sorted(scenario_scenes.items())
        },
        evaluable_scenario_counts=dict(sorted(evaluable_scenarios.items())),
        admitted_scenario_unique_scene_counts={
            scenario: len(scene_tokens)
            for scenario, scene_tokens in sorted(admitted_scenario_scenes.items())
        },
        admitted_pair_count=len(admitted_pairs),
        mini_admitted_pair_count=len(mini_admitted_pairs),
        traffic_status_sources=dict(sorted(status_sources.items())),
        admission_reasons=dict(sorted(admission_reasons.items())),
        admitted_keyframes=admitted,
        admitted_unique_scenes=len(admitted_scenes),
        admitted_keyframes_per_scene=_distribution(list(admitted_by_scene.values())),
        selected_eval_entries=selected_eval_entries,
        stationary_keyframes=stationary,
        scenes_with_stationary_keyframes=len(stationary_scenes),
        stationary_admitted_keyframes=stationary_admitted,
        stationary_displacement_threshold_m=stationary_displacement_threshold_m,
        qa_pair_counts=_distribution(qa_counts),
        key_object_counts=_distribution(key_object_counts),
        stop_line_geometry_keyframes=stop_line,
        ped_crossing_geometry_keyframes=ped_crossing,
        skipped_keyframes=skipped,
        notes=notes,
    )


def render_spike_report(
    stats: AvailabilityStats,
    *,
    eval_set_state: str = "draft",
    input_provenance: str | None = None,
    determinism_verified: bool = False,
) -> str:
    """将统计结果渲染为可版本管理的 Markdown。"""
    rows = [
        "# RegGround-AV trainval availability spike",
        "",
        f"- Total keyframes：{stats.total_keyframes or stats.boston_drivelm_keyframes}",
        f"- Unique scenes：{stats.unique_scenes}",
        f"- Admitted keyframes：{stats.admitted_keyframes}",
        f"- Admitted unique scenes：{stats.admitted_unique_scenes}",
        (
            "- 静止阈值：总位移 < "
            f"{stats.stationary_displacement_threshold_m:.2f} m"
        ),
        f"- 静止且已准入：{stats.stationary_admitted_keyframes}",
        (
            "- Stationary keyframe ratio："
            f"{stats.stationary_keyframes}/"
            f"{stats.total_keyframes or stats.boston_drivelm_keyframes}"
        ),
        (
            "- Scenes containing stationary keyframes："
            f"{stats.scenes_with_stationary_keyframes}/{stats.unique_scenes}"
        ),
        (
            "- Admitted keyframes per scene（min/median/p95/max）："
            f"{stats.admitted_keyframes_per_scene.minimum}/"
            f"{stats.admitted_keyframes_per_scene.median:.1f}/"
            f"{stats.admitted_keyframes_per_scene.p95}/"
            f"{stats.admitted_keyframes_per_scene.maximum}"
        ),
        *([f"- 输入基线：{input_provenance}"] if input_provenance else []),
        *(
            ["- 确定性验证：两次独立生成的报告和 eval_set 均逐字节一致。"]
            if determinism_verified else []
        ),
        "",
        "## 场景分层",
        "",
        (
            "| scenario_type | keyframes | unique scenes | admitted keyframes | "
            "admitted unique scenes |"
        ),
        "|---|---:|---:|---:|---:|",
    ]
    for scenario in sorted(
        set(stats.scenario_counts) | set(stats.evaluable_scenario_counts)
        | set(stats.admitted_scenario_unique_scene_counts),
    ):
        rows.append(
            f"| {scenario} | {stats.scenario_counts.get(scenario, 0)} "
            f"| {stats.scenario_unique_scene_counts.get(scenario, 0)} "
            f"| {stats.evaluable_scenario_counts.get(scenario, 0)} "
            f"| {stats.admitted_scenario_unique_scene_counts.get(scenario, 0)} |"
        )
    rows.extend([
        "",
        f"- 准入后 `(scene_token, scenario_type)` 配对总量：{stats.admitted_pair_count}",
        f"- mini 涉及的准入配对数：{stats.mini_admitted_pair_count}",
        f"- 正式评估集单元数：{len(stats.selected_eval_entries)}",
        "",
        "## Location 覆盖",
        "",
        "| location | DriveLM keyframes |",
        "|---|---:|",
    ])
    rows.extend(
        f"| {location} | {count} |"
        for location, count in stats.location_keyframes.items()
    )
    rows.extend([
        "",
        "## 灯态来源",
        "",
        "| traffic_light_status_source | keyframes | ratio |",
        "|---|---:|---:|",
    ])
    source_total = sum(stats.traffic_status_sources.values())
    rows.extend(
        f"| {source} | {count} | {count / source_total:.1%} |"
        for source, count in stats.traffic_status_sources.items()
    )
    none_count = stats.traffic_status_sources.get("none", 0)
    rows.extend([
        (
            "- `none` 占比："
            f"{none_count}/{source_total} ({none_count / source_total:.1%})"
            if source_total else "- `none` 占比：0/0 (n/a)"
        ),
        "- red_light 仅在 DriveLM 提及灯态的帧中定义；该分布限定论文的覆盖范围。",
        "",
        "## 上下文丰度",
        "",
        (
            "- QA pairs（min/median/p95/max）："
            f"{stats.qa_pair_counts.minimum}/{stats.qa_pair_counts.median:.1f}/"
            f"{stats.qa_pair_counts.p95}/{stats.qa_pair_counts.maximum}"
        ),
        (
            "- key objects（min/median/p95/max）："
            f"{stats.key_object_counts.minimum}/"
            f"{stats.key_object_counts.median:.1f}/"
            f"{stats.key_object_counts.p95}/{stats.key_object_counts.maximum}"
        ),
        "",
        "## 决策提示",
        "",
    ])
    rows.extend(
        f"- {layer}：{decision}"
        for layer, decision in THIN_LAYER_DECISIONS.items()
    )
    rows.extend(f"- {note}" for note in stats.notes)
    if not stats.notes:
        rows.append("- 当前分层样本量未触发 <20 薄层处置提示。")
    rows.append("")
    if stats.selected_eval_entries:
        eval_set_heading = (
            "## 正式评估集（已冻结）"
            if eval_set_state == "frozen"
            else "## 正式评估集候选（尚未冻结）"
        )
        rows.extend([
            eval_set_heading,
            "",
            "选择规则：horizon 内 ego 位移最大；平手按时间戳早者、再按 token 字典序。"
            "全程确定性，无随机成分（seed=42 未参与选择）。",
            "",
            "| token | scene | scenario | horizon 位移 (m) |",
            "|---|---|---|---:|",
            *[
                (
                    f"| {entry.frame_token} | {entry.scene_token} | {entry.scenario_type} "
                    f"| {entry.horizon_displacement_m:.3f} |"
                )
                for entry in stats.selected_eval_entries
            ],
        ])
    return "\n".join(rows) + "\n"


def select_representative_frames(
    frames: list[EvalSetEntry],
) -> list[EvalSetEntry]:
    """每个准入场景 × scenario 单元确定性选择接近帧。"""
    grouped: dict[tuple[str, str], list[EvalSetEntry]] = {}
    for frame in frames:
        grouped.setdefault((frame.scene_token, frame.scenario_type), []).append(frame)
    selected: list[EvalSetEntry] = []
    for group in sorted(grouped):
        selected.append(min(
            grouped[group],
            key=lambda frame: (
                -frame.horizon_displacement_m,
                frame.timestamp,
                frame.frame_token,
            ),
        ))
    return selected


def _ego_displacement_m(poses: list[dict[str, Any]]) -> float:
    """计算 ego 窗口首尾二维总位移。"""
    if len(poses) < 2:
        return 0.0
    start = poses[0].get("translation", [])
    end = poses[-1].get("translation", [])
    if len(start) < 2 or len(end) < 2:
        return 0.0
    return math.hypot(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))


def _frame_timestamp(raw_scene: Any) -> int:
    """返回 keyframe 时间戳，用于选择规则的稳定平手打破。"""
    timestamp = raw_scene.raw_json.get("sample_timestamp")
    if isinstance(timestamp, int) and timestamp >= 0:
        return timestamp
    if raw_scene.ego_poses:
        pose_timestamp = raw_scene.ego_poses[0].get("timestamp")
        if isinstance(pose_timestamp, int) and pose_timestamp >= 0:
            return pose_timestamp
    return 0


def _distribution(values: list[int]) -> CountDistribution:
    """计算离散计数的 min/median/p95/max。"""
    if not values:
        return CountDistribution()
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
    return CountDistribution(
        minimum=ordered[0],
        median=float(statistics.median(ordered)),
        p95=ordered[p95_index],
        maximum=ordered[-1],
    )


def _input_provenance(args: argparse.Namespace, stats: AvailabilityStats) -> str:
    """记录本次统计口径与上一版 metadata 覆盖口径的可解释差异。"""
    if args.prepared_input is not None:
        return (
            f"v1 以正式 runner 实际加载的 {args.prepared_input}（{stats.total_keyframes} 条）"
            "为唯一基准；上一版报告来自直接 metadata/DriveLM 覆盖（2,300 条）。"
            "已核验分类/准入代码与 f694f721 无差异，且生成已双跑逐字节一致：少 6 条是 "
            "prepared-input join 覆盖差异；约 20 条分层重分类（如 pedestrian −20、unknown +13）"
            "来自 facts 从 ad-hoc metadata 路径切换为生产 ingestion 输出。v2 消除了 v1 绕过正式"
            "输入路径的隐性规格违背；正式准入预测以 v2 数字为准。"
        )
    return (
        f"v1 以直接 metadata 输入 {args.drivelm_qa_path}（{stats.total_keyframes} 条）为基准；"
        "冻结与正式运行均由 classification_code_sha256 锁定同一分类/准入逻辑。"
    )


def _collect_from_args(
    args: argparse.Namespace,
    mini_scene_tokens: set[str],
) -> AvailabilityStats:
    """按 CLI 输入源执行一次完整统计。"""
    if args.prepared_input is not None:
        return collect_availability_stats_from_prepared_input(
            args.prepared_input,
            stationary_displacement_threshold_m=args.stationary_threshold_m,
            mini_scene_tokens=mini_scene_tokens,
        )
    if args.nuscenes_root is None or args.drivelm_qa_path is None:
        raise ValueError("metadata paths are required when prepared input is absent")
    return collect_availability_stats(
        args.nuscenes_root,
        args.drivelm_qa_path,
        stationary_displacement_threshold_m=args.stationary_threshold_m,
        mini_scene_tokens=mini_scene_tokens,
    )


def _build_eval_set(
    stats: AvailabilityStats,
    args: argparse.Namespace,
    mini_scene_tokens: set[str],
    *,
    state: Literal["draft", "frozen"],
    determinism_verified: bool,
) -> EvalSet:
    """将一次 spike 统计转成可冻结的五件套。"""
    source_path = args.prepared_input or args.drivelm_qa_path
    if source_path is None:
        raise ValueError("eval set requires an input source")
    return EvalSet(
        state=state,
        entries=stats.selected_eval_entries,
        mini_scene_tokens=sorted(mini_scene_tokens),
        generation_script="src/layer1/availability_spike.py",
        generation_commit=_git_commit(),
        spike_report=str(args.output),
        input_source=str(source_path),
        input_sha256=file_sha256(source_path),
        input_provenance=_input_provenance(args, stats),
        classification_source_files=classification_source_files(),
        classification_code_sha256=classification_code_sha256(),
        determinism_verified=determinism_verified,
    )


def main() -> None:
    """命令行 spike 入口。"""
    parser = argparse.ArgumentParser(description="统计 trainval 正式准入可得性")
    parser.add_argument("nuscenes_root", type=Path, nargs="?")
    parser.add_argument("drivelm_qa_path", type=Path, nargs="?")
    parser.add_argument(
        "--prepared-input",
        type=Path,
        default=None,
        help="正式 runner 使用的 RawScene JSON；指定后不读取 nuScenes metadata。",
    )
    parser.add_argument("--output", type=Path, default=Path("spike_report.md"))
    parser.add_argument("--stationary-threshold-m", type=float, default=1.0)
    parser.add_argument(
        "--mini-scenes",
        type=Path,
        default=Path("v1.0-mini/scene.json"),
        help="nuScenes mini 的 scene.json；正式选择前按场景排除。",
    )
    parser.add_argument(
        "--eval-set-output",
        type=Path,
        default=None,
        help="输出 eval_set_v1 JSON；默认写 draft，--freeze 后才可供正式 run 使用。",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="将输出标为 frozen；仅允许在 clean Git worktree 上执行。",
    )
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help="独立重跑一次，并要求报告与 eval_set 逐字节一致。",
    )
    args = parser.parse_args()
    mini_scene_tokens = set(load_mini_scene_tokens(args.mini_scenes))
    if args.freeze and args.eval_set_output is None:
        parser.error("--freeze requires --eval-set-output")
    if args.freeze and not args.verify_determinism:
        parser.error("--freeze requires --verify-determinism")
    if args.verify_determinism and args.eval_set_output is None:
        parser.error("--verify-determinism requires --eval-set-output")
    if args.freeze and _git_status_is_dirty():
        raise SystemExit("--freeze requires a clean Git worktree")
    if args.prepared_input is not None:
        if args.nuscenes_root is not None or args.drivelm_qa_path is not None:
            parser.error("--prepared-input cannot be combined with metadata positional paths")
    else:
        if args.nuscenes_root is None or args.drivelm_qa_path is None:
            parser.error("provide --prepared-input or both nuscenes_root and drivelm_qa_path")

    stats = _collect_from_args(args, mini_scene_tokens)
    eval_set_state: Literal["draft", "frozen"] = "frozen" if args.freeze else "draft"
    eval_set = _build_eval_set(
        stats,
        args,
        mini_scene_tokens,
        state=eval_set_state,
        determinism_verified=args.verify_determinism,
    )
    report = render_spike_report(
        stats,
        eval_set_state=eval_set_state,
        input_provenance=eval_set.input_provenance,
        determinism_verified=args.verify_determinism,
    )
    if args.verify_determinism:
        repeated_stats = _collect_from_args(args, mini_scene_tokens)
        repeated_eval_set = _build_eval_set(
            repeated_stats,
            args,
            mini_scene_tokens,
            state=eval_set_state,
            determinism_verified=True,
        )
        repeated_report = render_spike_report(
            repeated_stats,
            eval_set_state=eval_set_state,
            input_provenance=repeated_eval_set.input_provenance,
            determinism_verified=True,
        )
        if report != repeated_report or eval_set_json_bytes(eval_set) != eval_set_json_bytes(
            repeated_eval_set,
        ):
            raise SystemExit("determinism verification failed")
    args.output.write_text(report, encoding="utf-8")
    if args.eval_set_output is not None:
        write_eval_set(args.eval_set_output, eval_set)
    print(args.output)


def _git_commit() -> str:
    """返回生成脚本所在提交；非 Git 环境明确标为 unknown。"""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _git_status_is_dirty() -> bool:
    """冻结前确认生成代码对应一个真实提交状态。"""
    try:
        return bool(subprocess.run(
            ["git", "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return True


if __name__ == "__main__":
    main()
