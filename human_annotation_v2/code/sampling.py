"""盲标包 v2 的抽样规格。

抽样只读运行日志与重放上下文，不写任何生产目录。所有排序由
SHA-256(salt:frame_token:trajectory_id) 决定，seed 固定，因此同一输入
必然抽出同一批样本。

分层口径（对应用户抽样规格）：

* 主池 = LLM 裁定的 382 条候选。这 382 条在 benchmark 里全部是 ``exclude``，
  机器没有任何 ground truth，人工标注是唯一可得的判据；红灯 uncertain/cleared
  自然构成大头。
* predicate_human_agreement 层 = rule_engine 直裁且进入评分的 118 条中抽 30–40。
  规则引擎的判定不进 benchmark accuracy，只能靠人工核对谓词，故单列一层。
* 行人场景近全采：行人 LLM 候选全取，叠加谓词层覆盖，保证 12 个可评估行人场景
  全部出现。
* borderline（``pedestrian_gap_borderline`` / ``borderline_or_phase_horizon_unknown``）
  占比目标 30%，硬校验落在 25%–35%。
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from src.replay_context import ReplayedScene

SEED = 42

#: 谓词层未直裁的两类阈值带原因，构成 borderline 分层。
BORDERLINE_REASONS = frozenset(
    {"pedestrian_gap_borderline", "borderline_or_phase_horizon_unknown"}
)

CORE_TOTAL = 180
PREDICATE_QUOTA = 36
PREDICATE_BAND = (30, 40)
BORDERLINE_TARGET_RATIO = 0.30
BORDERLINE_BAND = (0.25, 0.35)
STRESS_TOTAL = 20
CALIBRATION_TOTAL = 8
PREFERENCE_SCENES = 36
PREFERENCE_BAND = (30, 40)


@dataclass(frozen=True)
class Candidate:
    """抽样框中的一个候选，含分层元数据与揭盲用真值。"""

    frame_token: str
    scene_id: str
    trajectory_id: str
    scenario: str
    decided_by: str
    system_status: str
    expected_verdict: str
    predicate_reason: str
    difficulty: str
    variant_type: str
    borderline: bool

    @property
    def key(self) -> tuple[str, str]:
        return (self.frame_token, self.trajectory_id)


@dataclass
class SampleRow:
    """抽中的一行，group 只写进 private sidecar。"""

    candidate: Candidate
    group: str
    stratum: str
    stress_kind: str = ""


@dataclass
class SamplingResult:
    core: list[SampleRow] = field(default_factory=list)
    stress: list[SampleRow] = field(default_factory=list)
    calibration: list[SampleRow] = field(default_factory=list)
    preference_frames: list[str] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def workbook_rows(self) -> list[SampleRow]:
        """正式表 = core + stress，按 P1-1 统一混洗后编号。"""
        rows = self.core + self.stress
        return order_by_hash(rows, salt="workbook", key=lambda row: row.candidate.key)


def stable_hash(*parts: str) -> str:
    payload = ":".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def order_by_hash(items: list[Any], *, salt: str, key: Any) -> list[Any]:
    """按 SHA-256(seed:salt:key) 升序确定性排序。"""
    return sorted(
        items,
        key=lambda item: stable_hash(str(SEED), salt, *key(item)),
    )


def largest_remainder(quota: int, weights: dict[str, int], *, floor: int = 0) -> dict[str, int]:
    """最大余数法分配名额；floor 保证非空层至少拿到 floor 个。

    分配结果按 key 字典序打破平手，不引入随机性。
    """
    live = {name: weight for name, weight in weights.items() if weight > 0}
    if not live:
        return {name: 0 for name in weights}
    if floor * len(live) > quota:
        raise ValueError(f"quota={quota} 不足以给 {len(live)} 个层各留 {floor} 个")

    allocation = {name: min(floor, live[name]) for name in live}
    remaining = quota - sum(allocation.values())
    total = sum(live.values())

    exact = {
        name: (live[name] - allocation[name]) * remaining / total if total else 0.0
        for name in live
    }
    base = {name: int(value) for name, value in exact.items()}
    for name in live:
        headroom = live[name] - allocation[name]
        base[name] = min(base[name], headroom)
    leftover = remaining - sum(base.values())

    order = sorted(
        live,
        key=lambda name: (-(exact[name] - int(exact[name])), name),
    )
    index = 0
    while leftover > 0:
        progressed = False
        for name in order:
            if leftover == 0:
                break
            if base[name] + allocation[name] < live[name]:
                base[name] += 1
                leftover -= 1
                progressed = True
        if not progressed:
            break
        index += 1
        if index > quota:
            break

    result = {name: allocation[name] + base[name] for name in live}
    for name in weights:
        result.setdefault(name, 0)
    return result


def build_frame(scenes: list[ReplayedScene]) -> list[Candidate]:
    """把重放结果展开成候选级抽样框。"""
    frame: list[Candidate] = []
    for scene in scenes:
        for trajectory_id in scene.trajectory_ids:
            veto = scene.verdict_by_id(trajectory_id)
            benchmark = scene.benchmark_by_id(trajectory_id)
            reason = str(benchmark.get("expected_verdict_reason") or "")
            frame.append(
                Candidate(
                    frame_token=scene.frame_token,
                    scene_id=scene.scene_id,
                    trajectory_id=trajectory_id,
                    scenario=scene.scenario_type,
                    decided_by=str(veto.get("decided_by")),
                    system_status=str(veto.get("status")),
                    expected_verdict=str(benchmark.get("expected_verdict")),
                    predicate_reason=reason,
                    difficulty=str(benchmark.get("difficulty")),
                    variant_type=str(benchmark.get("variant_type")),
                    borderline=reason in BORDERLINE_REASONS,
                )
            )
    return frame


def _take(pool: list[Candidate], count: int, *, salt: str) -> list[Candidate]:
    ordered = order_by_hash(pool, salt=salt, key=lambda item: item.key)
    return ordered[:count]


def sample_core(frame: list[Candidate]) -> tuple[list[SampleRow], dict[str, Any]]:
    """抽取主样本：LLM 主池 + 谓词层，满足行人近全采与 borderline 占比。"""
    llm_pool = [item for item in frame if item.decided_by == "llm"]
    predicate_pool = [
        item
        for item in frame
        if item.decided_by == "rule_engine" and item.expected_verdict != "exclude"
    ]

    rows: list[SampleRow] = []
    audit: dict[str, Any] = {
        "llm_pool_size": len(llm_pool),
        "predicate_pool_size": len(predicate_pool),
    }

    # ── 谓词层：按 (scenario, predicate_reason) 分层，小层保底 1 个 ──
    predicate_cells: dict[str, list[Candidate]] = {}
    for item in predicate_pool:
        predicate_cells.setdefault(
            f"{item.scenario}|{item.predicate_reason}", []
        ).append(item)
    predicate_alloc = largest_remainder(
        PREDICATE_QUOTA,
        {name: len(items) for name, items in predicate_cells.items()},
        floor=1,
    )
    for name in sorted(predicate_cells):
        for candidate in _take(
            predicate_cells[name], predicate_alloc[name], salt=f"predicate:{name}"
        ):
            rows.append(
                SampleRow(
                    candidate=candidate,
                    group="core_predicate",
                    stratum=f"predicate|{name}",
                )
            )
    audit["predicate_allocation"] = dict(sorted(predicate_alloc.items()))

    # ── LLM 层：行人近全采，其余按池规模比例分配 ──
    llm_quota = CORE_TOTAL - PREDICATE_QUOTA
    pedestrian = [item for item in llm_pool if item.scenario == "pedestrian"]
    for candidate in order_by_hash(
        pedestrian, salt="llm:pedestrian", key=lambda item: item.key
    ):
        rows.append(
            SampleRow(
                candidate=candidate,
                group="core_llm",
                stratum="llm|pedestrian|census",
            )
        )

    rest_quota = llm_quota - len(pedestrian)
    rest_pool = [item for item in llm_pool if item.scenario != "pedestrian"]
    scenario_weights = Counter(item.scenario for item in rest_pool)
    scenario_alloc = largest_remainder(rest_quota, dict(scenario_weights))
    audit["llm_scenario_allocation"] = {
        "pedestrian": len(pedestrian),
        **dict(sorted(scenario_alloc.items())),
    }

    # borderline 占比：行人普查已经贡献一部分，其余从各场景 borderline 池补足
    borderline_target = round(BORDERLINE_TARGET_RATIO * CORE_TOTAL)
    secured = sum(1 for item in pedestrian if item.borderline)
    needed = max(0, borderline_target - secured)

    borderline_capacity = {
        scenario: min(
            scenario_alloc[scenario],
            sum(1 for item in rest_pool if item.scenario == scenario and item.borderline),
        )
        for scenario in scenario_alloc
    }
    borderline_alloc = largest_remainder(
        min(needed, sum(borderline_capacity.values())),
        borderline_capacity,
    )
    audit["borderline_plan"] = {
        "target_total": borderline_target,
        "secured_by_pedestrian_census": secured,
        "needed_from_other_scenarios": needed,
        "allocation": dict(sorted(borderline_alloc.items())),
    }

    for scenario in sorted(scenario_alloc):
        for is_borderline in (True, False):
            count = (
                borderline_alloc[scenario]
                if is_borderline
                else scenario_alloc[scenario] - borderline_alloc[scenario]
            )
            if count <= 0:
                continue
            group_pool = [
                item
                for item in rest_pool
                if item.scenario == scenario and item.borderline is is_borderline
            ]
            # 场景内再按系统 status 分层，保持 uncertain/cleared 的自然大头
            status_cells: dict[str, list[Candidate]] = {}
            for item in group_pool:
                status_cells.setdefault(item.system_status, []).append(item)
            status_alloc = largest_remainder(
                count, {name: len(items) for name, items in status_cells.items()}
            )
            for status in sorted(status_cells):
                tag = "borderline" if is_borderline else "clear_cut"
                for candidate in _take(
                    status_cells[status],
                    status_alloc[status],
                    salt=f"llm:{scenario}:{tag}:{status}",
                ):
                    rows.append(
                        SampleRow(
                            candidate=candidate,
                            group="core_llm",
                            stratum=f"llm|{scenario}|{tag}|{status}",
                        )
                    )
    return rows, audit


def sample_stress(
    frame: list[Candidate],
    scenes: list[ReplayedScene],
    taken: set[tuple[str, str]],
) -> tuple[list[SampleRow], dict[str, Any]]:
    """抽取压力案例。类别按运行中实际存在的现象定义，不足不补。"""
    no_choosable = {
        scene.frame_token
        for scene in scenes
        if scene.record["final_label"].get("selection_outcome") == "no_choosable_candidate"
    }
    citation_repair = {
        scene.frame_token
        for scene in scenes
        if any(
            diagnostic.get("citation_failures")
            for diagnostic in scene.record["final_label"]["stage_diagnostics"]
        )
    }
    multi_condition = {
        scene.frame_token
        for scene in scenes
        if len(scene.record["layer2"].get("matched_conditions") or []) > 1
    }
    override = {
        scene.frame_token
        for scene in scenes
        if scene.record["layer2"].get("overrides_applied")
    }

    pools: dict[str, list[Candidate]] = {
        "no_choosable_uncertain": [
            item
            for item in frame
            if item.frame_token in no_choosable and item.system_status == "uncertain"
        ],
        "citation_repair": [
            item for item in frame if item.frame_token in citation_repair
        ],
        "llm_veto": [
            item
            for item in frame
            if item.decided_by == "llm" and item.system_status == "vetoed"
        ],
        "multi_condition_rule": [
            item for item in frame if item.frame_token in multi_condition
        ],
        "real_override": [item for item in frame if item.frame_token in override],
    }
    plan = {
        "no_choosable_uncertain": 6,
        "citation_repair": 5,
        "llm_veto": 5,
        "multi_condition_rule": 4,
        "real_override": 0,
    }

    rows: list[SampleRow] = []
    used = set(taken)
    availability: dict[str, int] = {}
    drawn: dict[str, int] = {}
    for kind in sorted(plan):
        pool = [item for item in pools[kind] if item.key not in used]
        availability[kind] = len(pools[kind])
        chosen = _take(pool, plan[kind], salt=f"stress:{kind}")
        drawn[kind] = len(chosen)
        for candidate in chosen:
            used.add(candidate.key)
            rows.append(
                SampleRow(
                    candidate=candidate,
                    group="stress",
                    stratum=f"stress|{kind}",
                    stress_kind=kind,
                )
            )

    return rows, {
        "planned": plan,
        "pool_size": availability,
        "drawn": drawn,
        "override_available": len(override),
        "note": (
            "v2 全量运行没有任何 override 生效，因此 real_override 类别为 0，"
            "不人为凑数；v1 的 scene-0286 不属于本次冻结输入。"
        ),
    }


def sample_calibration(
    frame: list[Candidate],
    scenes: list[ReplayedScene],
    taken: set[tuple[str, str]],
) -> tuple[list[SampleRow], dict[str, Any]]:
    """主样本之外抽校准轮，覆盖四类裁定分歧高发形态（P1-3）。"""
    features: dict[tuple[str, str], Any] = {}
    for scene in scenes:
        for trajectory_id in scene.trajectory_ids:
            features[(scene.frame_token, trajectory_id)] = scene.feature_by_id(
                trajectory_id
            )

    def is_creeping(candidate: Candidate) -> bool:
        feature = features[candidate.key]
        speed = feature.min_speed_in_zone_mps
        return bool(
            feature.entered_conflict_zone and speed is not None and 0.0 < speed <= 0.5
        )

    pools: dict[str, list[Candidate]] = {
        "obvious_violation": [
            item
            for item in frame
            if item.decided_by == "rule_engine" and item.expected_verdict == "vetoed"
        ],
        "obvious_pass": [
            item
            for item in frame
            if item.decided_by == "rule_engine" and item.expected_verdict == "cleared"
        ],
        "borderline_deceleration": [item for item in frame if item.borderline],
        "creeping": [item for item in frame if is_creeping(item)],
    }

    rows: list[SampleRow] = []
    used = set(taken)
    per_kind = CALIBRATION_TOTAL // len(pools)
    drawn: dict[str, int] = {}
    for kind in sorted(pools):
        pool = [item for item in pools[kind] if item.key not in used]
        chosen = _take(pool, per_kind, salt=f"calibration:{kind}")
        drawn[kind] = len(chosen)
        for candidate in chosen:
            used.add(candidate.key)
            rows.append(
                SampleRow(
                    candidate=candidate,
                    group="calibration",
                    stratum=f"calibration|{kind}",
                )
            )
    return rows, {"per_kind": per_kind, "drawn": drawn}


def sample_preference(scenes: list[ReplayedScene]) -> tuple[list[str], dict[str, Any]]:
    """抽场景级偏好表：优先纳入 ≥2 条候选被判 cleared 的场景（P0-1）。"""

    def cleared_count(scene: ReplayedScene) -> int:
        return sum(
            1
            for trajectory_id in scene.trajectory_ids
            if scene.verdict_by_id(trajectory_id)["status"] == "cleared"
        )

    rankable = [scene for scene in scenes if cleared_count(scene) >= 2]
    others = [scene for scene in scenes if cleared_count(scene) < 2]

    ordered = order_by_hash(
        rankable, salt="preference:rankable", key=lambda scene: (scene.frame_token,)
    )
    selected = ordered[:PREFERENCE_SCENES]
    if len(selected) < PREFERENCE_SCENES:
        filler = order_by_hash(
            others, salt="preference:filler", key=lambda scene: (scene.frame_token,)
        )
        selected += filler[: PREFERENCE_SCENES - len(selected)]

    return [scene.frame_token for scene in selected], {
        "rankable_scene_count": len(rankable),
        "selected": len(selected),
        "selected_rankable": sum(1 for scene in selected if cleared_count(scene) >= 2),
    }


def validate(result: SamplingResult, frame: list[Candidate]) -> dict[str, Any]:
    """发放前的硬校验。任何一条不满足都直接抛错，不允许"基本符合"。"""
    issues: list[str] = []
    core = result.core
    stress = result.stress
    calibration = result.calibration

    keys = [row.candidate.key for row in core + stress + calibration]
    if len(keys) != len(set(keys)):
        issues.append("core/stress/calibration 之间存在重复候选")

    if len(core) != CORE_TOTAL:
        issues.append(f"主样本数量 {len(core)} != {CORE_TOTAL}")

    predicate_rows = [row for row in core if row.group == "core_predicate"]
    low, high = PREDICATE_BAND
    if not low <= len(predicate_rows) <= high:
        issues.append(f"谓词层 {len(predicate_rows)} 不在 {PREDICATE_BAND}")
    if any(row.candidate.decided_by != "rule_engine" for row in predicate_rows):
        issues.append("谓词层混入非 rule_engine 裁定候选")

    llm_rows = [row for row in core if row.group == "core_llm"]
    if any(row.candidate.decided_by != "llm" for row in llm_rows):
        issues.append("LLM 层混入非 LLM 裁定候选")

    borderline = sum(1 for row in core if row.candidate.borderline)
    ratio = borderline / len(core) if core else 0.0
    low_ratio, high_ratio = BORDERLINE_BAND
    if not low_ratio <= ratio <= high_ratio:
        issues.append(f"borderline 占比 {ratio:.3f} 不在 {BORDERLINE_BAND}")

    pedestrian_scenes = {
        item.frame_token for item in frame if item.scenario == "pedestrian"
    }
    covered = {
        row.candidate.frame_token
        for row in core
        if row.candidate.scenario == "pedestrian"
    }
    if covered != pedestrian_scenes:
        issues.append(
            f"行人场景未近全采：覆盖 {len(covered)}/{len(pedestrian_scenes)}"
        )

    llm_pedestrian = [
        item for item in frame if item.decided_by == "llm" and item.scenario == "pedestrian"
    ]
    sampled_llm_pedestrian = [
        row for row in llm_rows if row.candidate.scenario == "pedestrian"
    ]
    if len(sampled_llm_pedestrian) != len(llm_pedestrian):
        issues.append("行人 LLM 候选未全取")

    # 压力案例允许少于计划值（本次 real_override 池为空，不凑数），
    # 但不允许超出计划值，否则说明分层逻辑串层。
    if len(stress) > STRESS_TOTAL:
        issues.append(f"压力案例 {len(stress)} 超出计划 {STRESS_TOTAL}")

    low_pref, high_pref = PREFERENCE_BAND
    if not low_pref <= len(result.preference_frames) <= high_pref:
        issues.append(f"偏好表场景数 {len(result.preference_frames)} 不在 {PREFERENCE_BAND}")

    if issues:
        raise ValueError("抽样校验未通过：\n  - " + "\n  - ".join(issues))

    return {
        "core_total": len(core),
        "stress_total": len(stress),
        "calibration_total": len(calibration),
        "workbook_rows": len(core) + len(stress),
        "predicate_layer": len(predicate_rows),
        "llm_layer": len(llm_rows),
        "borderline_count": borderline,
        "borderline_ratio": round(ratio, 4),
        "pedestrian_scenes_covered": len(covered),
        "pedestrian_scenes_total": len(pedestrian_scenes),
        "distinct_scenes_in_workbook": len(
            {row.candidate.frame_token for row in core + stress}
        ),
        "preference_scenes": len(result.preference_frames),
    }


def run_sampling(scenes: list[ReplayedScene]) -> tuple[SamplingResult, list[Candidate]]:
    """执行完整抽样流程并返回结果与抽样框。"""
    frame = build_frame(scenes)
    core, core_audit = sample_core(frame)
    taken = {row.candidate.key for row in core}
    stress, stress_audit = sample_stress(frame, scenes, taken)
    taken |= {row.candidate.key for row in stress}
    calibration, calibration_audit = sample_calibration(frame, scenes, taken)
    preference_frames, preference_audit = sample_preference(scenes)

    result = SamplingResult(
        core=core,
        stress=stress,
        calibration=calibration,
        preference_frames=preference_frames,
    )
    summary = validate(result, frame)
    result.manifest = {
        "seed": SEED,
        "frame_size": len(frame),
        "core": core_audit,
        "stress": stress_audit,
        "calibration": calibration_audit,
        "preference": preference_audit,
        "summary": summary,
    }
    return result, frame


__all__ = [
    "BORDERLINE_BAND",
    "BORDERLINE_REASONS",
    "CORE_TOTAL",
    "Candidate",
    "SampleRow",
    "SamplingResult",
    "build_frame",
    "largest_remainder",
    "order_by_hash",
    "run_sampling",
    "stable_hash",
    "validate",
]
