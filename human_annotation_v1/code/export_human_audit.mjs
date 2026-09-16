import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";

import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

import { loadRuleCards } from "./annotation_schema.mjs";

const DEFAULTS = {
  input: "../logs/eval_set_v1_full_v3_5.jsonl",
  outputRoot: ".",
  seed: 42,
  ruleGraph: "../data/layer2/rule_graph.yaml",
  referenceCommit: "03c4701",
};

const CORE_LLM_QUOTA_PROFILES = {
  legacy_all_scenarios: {
    pedestrian: 49,
    oncoming: 21,
    red_light: 13,
    yellow_light: 5,
    emergency: 2,
  },
  v2_mvp: {
    pedestrian: 53,
    oncoming: 23,
    red_light: 14,
  },
};

const BORDERLINE_VERDICT_SAMPLING = {
  reasons: [
    "gap_borderline",
    "pedestrian_gap_borderline",
    "borderline_or_phase_horizon_unknown",
  ],
  targetRatio: 0.30,
  minRatio: 0.25,
  maxRatio: 0.35,
};

const FORCED_STRESS_CASE = {
  sceneName: "scene-0286",
  frameToken: "297cef22e0f143ceb1e4ef93088bcfb2",
  trajectoryId: "traj_a",
};

const BLIND_HEADERS = [
  "audit_id",
  "scenario_type",
  "scene_description",
  "structured_facts",
  "retrieved_rules",
  "override_relations",
  "trajectory_id",
  "trajectory_summary",
  "human_verdict",
  "human_applicable_rule_ids",
  "human_violated_rule_ids",
  "human_confidence",
  "human_reason",
  "notes",
];

function parseArgs(argv) {
  const options = { ...DEFAULTS };
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    const value = argv[index + 1];
    if (key === "--input") options.input = value;
    else if (key === "--output-root") options.outputRoot = value;
    else if (key === "--seed") options.seed = Number(value);
    else if (key === "--rule-graph") options.ruleGraph = value;
    else if (key === "--reference-commit") options.referenceCommit = value;
    else throw new Error(`未知参数: ${key}`);
    index += 1;
  }
  if (!Number.isInteger(options.seed)) throw new Error("--seed 必须是整数");
  return options;
}

function sha256Text(value) {
  return crypto.createHash("sha256").update(value).digest("hex");
}

async function sha256File(filePath) {
  return sha256Text(await fs.readFile(filePath));
}

function candidateKey(candidate) {
  return `${candidate.frameToken}:${candidate.trajectoryId}`;
}

function rank(candidate, salt, seed) {
  return sha256Text(`${salt}:${seed}:${candidateKey(candidate)}`);
}

function isBorderlineCandidate(candidate) {
  const reason = candidate.benchmark?.expected_verdict_reason;
  return BORDERLINE_VERDICT_SAMPLING.reasons.includes(reason);
}

function quotaProfile(candidates) {
  const scenarios = new Set(candidates.map((candidate) => candidate.scenarioType));
  const v2Scenarios = new Set(["pedestrian", "oncoming", "red_light"]);
  if ([...scenarios].every((scenario) => v2Scenarios.has(scenario))) {
    return {
      name: "v2_mvp",
      llmQuotas: CORE_LLM_QUOTA_PROFILES.v2_mvp,
    };
  }
  return {
    name: "legacy_all_scenarios",
    llmQuotas: CORE_LLM_QUOTA_PROFILES.legacy_all_scenarios,
  };
}

function coreStratum(candidate, llmQuotas) {
  if (candidate.source === "rule_engine") return "rule_engine";
  if (candidate.source === "llm" && candidate.scenarioType in llmQuotas) {
    return `llm_${candidate.scenarioType}`;
  }
  return null;
}

function booleanText(value) {
  if (value === true) return "是";
  if (value === false) return "否";
  return "未知";
}

function summarizeFacts(facts) {
  const zones = Array.isArray(facts.conflict_zones) ? facts.conflict_zones : [];
  const zoneKinds = [...new Set(zones.map((zone) => zone.kind))].sort();
  const nearest = zones
    .map((zone) => zone.distance_from_ego_m)
    .filter((value) => typeof value === "number")
    .sort((left, right) => left - right)[0];
  return [
    `红灯=${booleanText(facts.has_red_light)}`,
    `黄灯=${booleanText(facts.has_yellow_light)}`,
    `灯态来源=${facts.traffic_light_status_source ?? "未知"}`,
    `行人在横道内=${booleanText(facts.pedestrian_in_crosswalk)}`,
    `对向车辆运动=${booleanText(facts.oncoming_vehicle_moving)}`,
    `特种车辆执行任务=${booleanText(facts.emergency_vehicle_active)}`,
    `自车位于支路=${booleanText(facts.ego_on_minor_road)}`,
    `路口场景=${booleanText(facts.location_is_intersection)}`,
    `horizon位移=${Number(facts.ego_displacement_m ?? 0).toFixed(1)}m`,
    `冲突区=${zoneKinds.length ? zoneKinds.join("/") : "无"}(${zones.length})`,
    `最近冲突区距离=${nearest === undefined ? "未知" : `${nearest.toFixed(1)}m`}`,
  ].join("；");
}

function summarizeRules(rules) {
  if (!Array.isArray(rules) || rules.length === 0) return "无召回规则";
  return rules
    .map((rule) => {
      const status = rule.status ?? "active";
      return `${rule.node_id} [${rule.severity}/${status}] ${rule.description}`;
    })
    .join("\n");
}

function summarizeOverrides(overrides) {
  if (!Array.isArray(overrides) || overrides.length === 0) return "无";
  return overrides
    .map(
      (item) =>
        `${item.overriding_rule_id} 覆盖 ${item.overridden_rule_id}` +
        `（条件 ${item.condition_id}；${item.reason}）`,
    )
    .join("\n");
}

function buildCandidates(records) {
  const candidates = [];
  for (const record of records) {
    const context = record.layer1.context;
    const ids = context.candidate_trajectory_ids;
    const features = context.trajectory_features;
    const featureById = new Map(ids.map((id, index) => [id, features[index]]));
    const benchmarks = record.layer1.benchmark_labels_debug_only?.labels ?? [];
    const benchmarkById = new Map(benchmarks.map((item) => [item.anonymous_id, item]));
    for (const verdictEntry of record.final_label.verdicts) {
      const trajectoryId = verdictEntry.trajectory_id;
      const veto = verdictEntry.veto;
      const feature = featureById.get(trajectoryId);
      if (!feature) throw new Error(`缺少轨迹特征: ${record.index}/${trajectoryId}`);
      candidates.push({
        record,
        index: record.index,
        sceneId: record.raw_scene.scene_id,
        sceneName: record.raw_scene.scene_name,
        frameToken: record.raw_scene.frame_token,
        scenarioType: context.scenario_type,
        trajectoryId,
        feature,
        source: veto.decided_by,
        systemVerdict: veto.status,
        systemReason: veto.reason,
        systemRuleIds: veto.violated_rule_ids ?? [],
        benchmark: benchmarkById.get(trajectoryId) ?? null,
      });
    }
  }
  return candidates;
}

function pickUniqueScenes(pool, count, usedScenes, salt, seed) {
  const bestByScene = new Map();
  for (const candidate of pool) {
    if (usedScenes.has(candidate.sceneId)) continue;
    const current = bestByScene.get(candidate.sceneId);
    if (!current || rank(candidate, `${salt}:candidate`, seed) < rank(current, `${salt}:candidate`, seed)) {
      bestByScene.set(candidate.sceneId, candidate);
    }
  }
  const ordered = [...bestByScene.values()].sort((left, right) =>
    rank(left, `${salt}:scene`, seed).localeCompare(rank(right, `${salt}:scene`, seed)),
  );
  if (ordered.length < count) {
    throw new Error(`${salt} 可用独立场景 ${ordered.length}，不足 ${count}`);
  }
  const selected = ordered.slice(0, count);
  selected.forEach((candidate) => usedScenes.add(candidate.sceneId));
  return selected;
}

function selectCore(candidates, seed, llmQuotas) {
  const forcedScene = candidates.find(
    (candidate) =>
      candidate.sceneName === FORCED_STRESS_CASE.sceneName &&
      candidate.frameToken === FORCED_STRESS_CASE.frameToken &&
      candidate.trajectoryId === FORCED_STRESS_CASE.trajectoryId,
  )?.sceneId;
  if (!forcedScene) throw new Error("未找到强制压力案例 scene-0286/traj_a");

  const usedScenes = new Set([forcedScene]);
  const selected = [];

  const ruleEngine = pickUniqueScenes(
    candidates.filter((candidate) => candidate.source === "rule_engine"),
    30,
    usedScenes,
    "core:rule_engine",
    seed,
  );
  selected.push(...ruleEngine.map((candidate) => ({ ...candidate, group: "core", stratum: "rule_engine" })));

  const scenarioOrder = [
    "red_light",
    "emergency",
    "yellow_light",
    "oncoming",
    "pedestrian",
  ].filter((scenario) => scenario in llmQuotas);
  for (const scenario of scenarioOrder) {
    const quota = llmQuotas[scenario];
    const sample = pickUniqueScenes(
      candidates.filter(
        (candidate) => candidate.source === "llm" && candidate.scenarioType === scenario,
      ),
      quota,
      usedScenes,
      `core:llm:${scenario}`,
      seed,
    );
    selected.push(
      ...sample.map((candidate) => ({ ...candidate, group: "core", stratum: `llm_${scenario}` })),
    );
  }
  return selected;
}

function pickStress(pool, count, selectedKeys, usedStressScenes, salt, seed) {
  const filtered = pool.filter(
    (candidate) => !selectedKeys.has(candidateKey(candidate)) && !usedStressScenes.has(candidate.sceneId),
  );
  const picked = pickUniqueScenes(filtered, count, usedStressScenes, salt, seed);
  picked.forEach((candidate) => selectedKeys.add(candidateKey(candidate)));
  return picked;
}

function selectStress(candidates, core, seed) {
  const selectedKeys = new Set(core.map(candidateKey));
  const usedStressScenes = new Set();
  const forced = candidates.find(
    (candidate) =>
      candidate.sceneName === FORCED_STRESS_CASE.sceneName &&
      candidate.frameToken === FORCED_STRESS_CASE.frameToken &&
      candidate.trajectoryId === FORCED_STRESS_CASE.trajectoryId,
  );
  if (!forced) throw new Error("未找到强制压力案例 scene-0286/traj_a");
  selectedKeys.add(candidateKey(forced));
  usedStressScenes.add(forced.sceneId);

  const stress = [{ ...forced, group: "stress", stratum: "stress_override" }];
  const multiConditionPool = candidates.filter(
    (candidate) =>
      candidate.source === "llm" &&
      (candidate.record.layer2.matched_conditions ?? []).length >= 2,
  );
  stress.push(
    ...pickStress(
      multiConditionPool,
      4,
      selectedKeys,
      usedStressScenes,
      "stress:multi_condition",
      seed,
    ).map((candidate) => ({
      ...candidate,
      group: "stress",
      stratum: "stress_multi_condition",
    })),
  );

  const rareVetoPool = candidates.filter(
    (candidate) => candidate.source === "llm" && candidate.systemVerdict === "vetoed",
  );
  stress.push(
    ...pickStress(rareVetoPool, 5, selectedKeys, usedStressScenes, "stress:llm_veto", seed).map(
      (candidate) => ({ ...candidate, group: "stress", stratum: "stress_llm_veto" }),
    ),
  );

  const noChoosablePool = candidates.filter(
    (candidate) =>
      candidate.source === "llm" &&
      candidate.systemVerdict === "uncertain" &&
      candidate.record.final_label.selection_outcome === "no_choosable_candidate",
  );
  stress.push(
    ...pickStress(
      noChoosablePool,
      5,
      selectedKeys,
      usedStressScenes,
      "stress:no_choosable",
      seed,
    ).map((candidate) => ({ ...candidate, group: "stress", stratum: "stress_no_choosable" })),
  );

  const citationRepairPool = candidates.filter(
    (candidate) =>
      candidate.source === "llm" &&
      (candidate.record.final_label.partial_reasons ?? []).includes("citation_repair"),
  );
  stress.push(
    ...pickStress(
      citationRepairPool,
      5,
      selectedKeys,
      usedStressScenes,
      "stress:citation_repair",
      seed,
    ).map((candidate) => ({ ...candidate, group: "stress", stratum: "stress_citation_repair" })),
  );
  return stress;
}

function enforceBorderlineVerdictShare(candidates, core, stress, seed, llmQuotas) {
  if (!candidates.some(isBorderlineCandidate)) return core;

  const target = Math.round(
    (core.length + stress.length) * BORDERLINE_VERDICT_SAMPLING.targetRatio,
  );
  const selectedKeys = new Set([...core, ...stress].map(candidateKey));
  const usedCoreScenes = new Set(core.map((candidate) => candidate.sceneId));
  const result = [...core];
  let current = [...core, ...stress].filter(isBorderlineCandidate).length;

  while (current !== target) {
    const needBorderline = current < target;
    const outgoingOrder = result
      .map((candidate, index) => ({ candidate, index }))
      .filter(({ candidate }) => isBorderlineCandidate(candidate) !== needBorderline)
      .sort((left, right) => rank(left.candidate, "borderline:outgoing", seed)
        .localeCompare(rank(right.candidate, "borderline:outgoing", seed)));
    const replacementPool = candidates
      .filter((candidate) => isBorderlineCandidate(candidate) === needBorderline)
      .sort((left, right) => rank(left, "borderline:replacement", seed)
        .localeCompare(rank(right, "borderline:replacement", seed)));

    let replaced = false;
    for (const { candidate: outgoing, index } of outgoingOrder) {
      const stratum = outgoing.stratum;
      const replacement = replacementPool.find((candidate) =>
        coreStratum(candidate, llmQuotas) === stratum
        && !selectedKeys.has(candidateKey(candidate))
        && (!usedCoreScenes.has(candidate.sceneId) || candidate.sceneId === outgoing.sceneId));
      if (!replacement) continue;

      selectedKeys.delete(candidateKey(outgoing));
      usedCoreScenes.delete(outgoing.sceneId);
      const selected = { ...replacement, group: "core", stratum };
      result[index] = selected;
      selectedKeys.add(candidateKey(selected));
      usedCoreScenes.add(selected.sceneId);
      current += needBorderline ? 1 : -1;
      replaced = true;
      break;
    }
    if (!replaced) {
      throw new Error(`borderline 候选无法达到目标 ${target}/${core.length + stress.length}`);
    }
  }
  return result;
}

function assignAuditIds(samples, seed) {
  return [...samples]
    .sort((left, right) => rank(left, "blind_order", seed).localeCompare(rank(right, "blind_order", seed)))
    .map((sample, index) => ({ ...sample, auditId: `audit_${String(index + 1).padStart(3, "0")}` }));
}

function toBlind(sample) {
  const record = sample.record;
  const description =
    record.raw_scene.drivelm_scene_description ||
    record.raw_scene.scene_description ||
    `关键词：${(record.layer1.context.description?.keywords ?? []).join("、")}`;
  return {
    audit_id: sample.auditId,
    scenario_type: sample.scenarioType,
    scene_description: description,
    structured_facts: summarizeFacts(record.layer1.context.scene_facts),
    retrieved_rules: summarizeRules(record.layer2.rules),
    override_relations: summarizeOverrides(record.layer2.overrides_applied),
    trajectory_id: sample.trajectoryId,
    trajectory_summary: sample.feature.natural_language_summary,
    human_verdict: "",
    human_applicable_rule_ids: "",
    human_violated_rule_ids: "",
    human_confidence: "",
    human_reason: "",
    notes: "",
  };
}

function toSidecar(sample) {
  const benchmark = sample.benchmark;
  return {
    audit_id: sample.auditId,
    sample_group: sample.group,
    stratum: sample.stratum,
    index: sample.index,
    scene_id: sample.sceneId,
    scene_name: sample.sceneName,
    frame_token: sample.frameToken,
    scenario_type: sample.scenarioType,
    trajectory_id: sample.trajectoryId,
    system_verdict: sample.systemVerdict,
    decided_by: sample.source,
    system_reason: sample.systemReason,
    system_rule_ids: sample.systemRuleIds,
    system_chosen: sample.record.final_label.chosen_trajectory_id === sample.trajectoryId,
    selection_outcome: sample.record.final_label.selection_outcome,
    final_status: sample.record.final_label.status,
    partial_reasons: sample.record.final_label.partial_reasons ?? [],
    benchmark_expected_verdict: benchmark?.expected_verdict ?? null,
    benchmark_variant_type: benchmark?.variant_type ?? null,
    benchmark_difficulty: benchmark?.difficulty ?? null,
    benchmark_reason: benchmark?.expected_verdict_reason ?? null,
    sampling_band: isBorderlineCandidate(sample) ? "borderline_medium" : "standard",
    sample_hash: sha256Text(`${sample.group}:${sample.stratum}:${candidateKey(sample)}`),
  };
}

function groupCandidatesByFrame(candidates, frameTokens) {
  const groups = new Map();
  for (const candidate of candidates) {
    if (!frameTokens.has(candidate.frameToken)) continue;
    if (!groups.has(candidate.frameToken)) {
      groups.set(candidate.frameToken, {
        frameToken: candidate.frameToken,
        sceneId: candidate.sceneId,
        scenarioType: candidate.scenarioType,
        record: candidate.record,
        candidates: [],
      });
    }
    groups.get(candidate.frameToken).candidates.push(candidate);
  }
  return [...groups.values()];
}

function selectPreferenceScenes(candidates, core, seed) {
  const coreFrames = new Set(core.map((sample) => sample.frameToken));
  const groups = groupCandidatesByFrame(candidates, coreFrames);
  const scoreable = groups.filter(
    (group) => group.candidates.filter((candidate) => candidate.systemVerdict === "cleared").length >= 2,
  );
  const fallback = groups.filter((group) => !scoreable.includes(group));
  const eligible = scoreable.length >= 30
    ? scoreable
    : [...scoreable, ...fallback.slice(0, 30 - scoreable.length)];
  const selected = eligible
    .sort((left, right) => rank(left.candidates[0], "preference_scene", seed)
      .localeCompare(rank(right.candidates[0], "preference_scene", seed)))
    .slice(0, 40);
  return selected.map((group, index) => ({
    ...group,
    preferenceId: `pref_${String(index + 1).padStart(3, "0")}`,
    scoreable: scoreable.includes(group),
    candidates: [...group.candidates].sort((left, right) =>
      rank(left, "preference_candidate_order", seed)
        .localeCompare(rank(right, "preference_candidate_order", seed))),
  }));
}

function preferenceHeaders(maxCandidates) {
  const headers = [
    "preference_id",
    "scenario_type",
    "scene_description",
    "structured_facts",
    "retrieved_rules",
    "override_relations",
  ];
  for (let index = 1; index <= maxCandidates; index += 1) {
    headers.push(`candidate_${index}_id`, `candidate_${index}_summary`);
  }
  headers.push("human_best_trajectory", "human_ranking", "human_confidence", "human_reason", "notes");
  return headers;
}

function toPreferenceRow(group, headers) {
  const record = group.record;
  const description = record.raw_scene.drivelm_scene_description
    || record.raw_scene.scene_description
    || `关键词：${(record.layer1.context.description?.keywords ?? []).join("、")}`;
  const row = {
    preference_id: group.preferenceId,
    scenario_type: group.scenarioType,
    scene_description: description,
    structured_facts: summarizeFacts(record.layer1.context.scene_facts),
    retrieved_rules: summarizeRules(record.layer2.rules),
    override_relations: summarizeOverrides(record.layer2.overrides_applied),
    human_best_trajectory: "",
    human_ranking: "",
    human_confidence: "",
    human_reason: "",
    notes: "",
  };
  group.candidates.forEach((candidate, index) => {
    row[`candidate_${index + 1}_id`] = candidate.trajectoryId;
    row[`candidate_${index + 1}_summary`] = candidate.feature.natural_language_summary;
  });
  return headers.map((header) => row[header] ?? "");
}

function toPreferenceObject(group, headers) {
  const values = toPreferenceRow(group, headers);
  return Object.fromEntries(headers.map((header, index) => [header, values[index]]));
}

function calibrationKind(feature) {
  const minSpeed = Number(feature.min_speed_in_zone_mps);
  const minDistance = Number(feature.min_distance_to_conflict_m);
  if (feature.entered_conflict_zone && Number.isFinite(minSpeed) && minSpeed < 1) return "creeping";
  if (feature.entered_conflict_zone && Number.isFinite(minSpeed) && minSpeed < 3) {
    return "critical_deceleration";
  }
  if (feature.entered_conflict_zone && Number.isFinite(minSpeed) && minSpeed >= 3) {
    return "obvious_violation";
  }
  if (!feature.entered_conflict_zone && (feature.stopped_before_zone || minDistance > 0)) {
    return "obvious_clear";
  }
  return null;
}

function selectCalibrationSamples(candidates, core, seed) {
  const coreFrames = new Set(core.map((sample) => sample.frameToken));
  const coreScenes = new Set(core.map((sample) => sample.sceneId));
  const pools = new Map(["critical_deceleration", "creeping", "obvious_violation", "obvious_clear"].map(
    (kind) => [kind, candidates.filter((candidate) =>
      !coreFrames.has(candidate.frameToken)
      && !coreScenes.has(candidate.sceneId)
      && calibrationKind(candidate.feature) === kind)],
  ));
  const selected = [];
  const usedFrames = new Set();
  for (const kind of pools.keys()) {
    const pool = pools.get(kind).sort((left, right) =>
      rank(left, `calibration:${kind}`, seed).localeCompare(rank(right, `calibration:${kind}`, seed)));
    const candidate = pool.find((item) => !usedFrames.has(item.frameToken));
    if (!candidate) throw new Error(`校准轮缺少类别：${kind}`);
    selected.push({ ...candidate, calibrationKind: kind });
    usedFrames.add(candidate.frameToken);
  }
  for (const kind of pools.keys()) {
    if (selected.filter((sample) => sample.calibrationKind === kind).length >= 2) continue;
    const pool = pools.get(kind);
    const candidate = pool.find((item) => !usedFrames.has(item.frameToken));
    if (candidate) {
      selected.push({ ...candidate, calibrationKind: kind });
      usedFrames.add(candidate.frameToken);
    }
  }
  const fallback = candidates
    .filter((candidate) =>
      !coreFrames.has(candidate.frameToken)
      && !coreScenes.has(candidate.sceneId)
      && !usedFrames.has(candidate.frameToken))
    .sort((left, right) => rank(left, "calibration:fallback", seed)
      .localeCompare(rank(right, "calibration:fallback", seed)));
  for (const candidate of fallback) {
    if (selected.length >= 8) break;
    selected.push({ ...candidate, calibrationKind: calibrationKind(candidate.feature) ?? "mixed" });
    usedFrames.add(candidate.frameToken);
  }
  if (selected.length !== 8) throw new Error(`校准轮应生成8行，实际为 ${selected.length}`);
  return selected.map((sample, index) => ({ ...sample, auditId: `cal_${String(index + 1).padStart(3, "0")}` }));
}

function toCalibrationSidecar(sample) {
  return {
    calibration_id: sample.auditId,
    category: sample.calibrationKind,
    scene_id: sample.sceneId,
    frame_token: sample.frameToken,
    trajectory_id: sample.trajectoryId,
  };
}

function countBy(items, keyFn) {
  const counts = {};
  for (const item of items) {
    const key = keyFn(item);
    counts[key] = (counts[key] ?? 0) + 1;
  }
  return Object.fromEntries(Object.entries(counts).sort(([left], [right]) => left.localeCompare(right)));
}

function verify(samples, blindRows, llmQuotas) {
  const core = samples.filter((sample) => sample.group === "core");
  const stress = samples.filter((sample) => sample.group === "stress");
  if (core.length !== 120) throw new Error(`主样本数量错误: ${core.length}`);
  if (stress.length !== 20) throw new Error(`压力样本数量错误: ${stress.length}`);
  if (new Set(core.map((sample) => sample.sceneId)).size !== 120) {
    throw new Error("主样本未满足一场景一候选");
  }
  if (new Set(samples.map(candidateKey)).size !== samples.length) throw new Error("存在重复候选");
  if (new Set(samples.map((sample) => sample.auditId)).size !== samples.length) {
    throw new Error("audit_id 不唯一");
  }
  const coreStrata = countBy(core, (sample) => sample.stratum);
  const expected = { rule_engine: 30 };
  for (const [scenario, count] of Object.entries(llmQuotas)) {
    expected[`llm_${scenario}`] = count;
  }
  for (const [stratum, count] of Object.entries(expected)) {
    if (coreStrata[stratum] !== count) {
      throw new Error(`分层数量错误 ${stratum}: ${coreStrata[stratum]} != ${count}`);
    }
  }
  const forced = samples.find(
    (sample) =>
      sample.sceneName === FORCED_STRESS_CASE.sceneName &&
      sample.frameToken === FORCED_STRESS_CASE.frameToken &&
      sample.trajectoryId === FORCED_STRESS_CASE.trajectoryId,
  );
  if (!forced || forced.stratum !== "stress_override") throw new Error("强制压力案例未纳入");
  for (const row of blindRows) {
    if (Object.keys(row).join("|") !== BLIND_HEADERS.join("|")) {
      throw new Error(`盲标列异常: ${row.audit_id}`);
    }
  }
  if (samples.some(isBorderlineCandidate)) {
    const borderlineRatio = samples.filter(isBorderlineCandidate).length / samples.length;
    if (
      borderlineRatio < BORDERLINE_VERDICT_SAMPLING.minRatio
      || borderlineRatio > BORDERLINE_VERDICT_SAMPLING.maxRatio
    ) {
      throw new Error(`borderline 候选占比越界: ${borderlineRatio.toFixed(4)}`);
    }
  }
}

function verifyPreferenceGroups(groups) {
  if (groups.length < 30 || groups.length > 40) {
    throw new Error(`偏好场景数量必须在30–40之间，实际为 ${groups.length}`);
  }
  if (groups.some((group) => group.candidates.length < 2)) {
    throw new Error("偏好表每个场景至少需要两条候选轨迹");
  }
  if (new Set(groups.map((group) => group.preferenceId)).size !== groups.length) {
    throw new Error("preference_id 不唯一");
  }
}

function columnName(number) {
  let value = number;
  let result = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    result = String.fromCharCode(65 + remainder) + result;
    value = Math.floor((value - 1) / 26);
  }
  return result;
}

function instructionRows(ruleCards, { title, calibration = false, preference = false }) {
  const rows = [
    [title, "请先阅读说明；每名标注者使用独立副本。"],
    ["隔离纪律", "只依据本工作簿证据独立判断；不要查看原始日志、private sidecar、系统结果或 benchmark。"],
    ["场景描述说明", "scene_description 为数据集原文，仅作场景背景；判定证据以 structured_facts 和 trajectory_summary 为准。"],
  ];
  if (preference) {
    rows.push(
      ["human_best_trajectory", "从当前场景全部候选中选择一条最优轨迹；若没有可接受轨迹填 NONE。"],
      ["human_ranking", "填写全部候选的从优到劣排序，例如 traj_c>traj_a>traj_e；允许并列，例如 traj_a=traj_b>traj_c。"],
      ["偏好规则", "只比较当前行列出的候选，不根据候选 ID 字母或呈现顺序推断答案。"],
    );
  } else {
    rows.push(
      ["human_verdict", "cleared=无明确硬规则否决；vetoed=明确违反硬规则；uncertain=信息不足或冲突。"],
      ["human_applicable_rule_ids", "填写当前场景和判断相关的规则 ID；多个用英文分号分隔；无规则填 NONE。"],
      ["human_violated_rule_ids", "只填确认违反的规则 ID；cleared/uncertain 填 NONE；vetoed 至少填一条 hard rule。"],
      ["human_confidence", "填写 1–5 的整数；1=很不确定，5=很确定。"],
      ["human_reason", "用一句完整的话说明事实、轨迹行为与规则之间的关系。"],
      ["漏检规则", "若认为某条规则适用于当前场景但未出现在 retrieved_rules 列，请照规则卡填入 human_applicable_rule_ids，并在 notes 注明“未检索到”。"],
    );
  }
  rows.push(["全量规则卡", "以下为本次规则图谱中的全部合法规则；rule ID 请按原样填写。"]);
  rows.push(...ruleCards.map((card) => [
    card.ruleId,
    `${card.severity}｜${card.description}`,
  ]));
  rows.push([
    "示例（虚构，不计入）",
    preference
      ? "human_best_trajectory=traj_b；human_ranking=traj_b>traj_a=traj_c；human_confidence=4；human_reason=traj_b在冲突区前完成让行且保持安全间距。"
      : "human_verdict=vetoed；human_applicable_rule_ids=R-YLD-01；human_violated_rule_ids=R-YLD-01；human_confidence=4；human_reason=轨迹进入有人行人的冲突区且未让行；notes=示例,勿计入。",
  ]);
  if (calibration) {
    rows.push(["校准轮", "本表仅用于正式发放前校准，不计入正式样本；请两名标注者独立完成后讨论分歧。"]);
  } else {
    rows.push(["校准轮", "正式发放前须先完成独立校准轮；将实际分歧和裁定标准记录在项目负责人版本的说明页后再发放。"]);
  }
  return rows;
}

function styleInstructionSheet(sheet, rows) {
  sheet.showGridLines = false;
  const lastRow = rows.length;
  sheet.getRange(`A1:B${lastRow}`).values = rows;
  sheet.getRange("A1:B1").format = {
    fill: "#1F4E78",
    font: { bold: true, color: "#FFFFFF", size: 14 },
  };
  sheet.getRange(`A2:A${lastRow}`).format = {
    fill: "#D9EAF7",
    font: { bold: true, color: "#17365D" },
  };
  sheet.getRange(`A1:B${lastRow}`).format.wrapText = true;
  sheet.getRange(`A1:B${lastRow}`).format.borders = {
    preset: "inside",
    style: "thin",
    color: "#D9E2F3",
  };
  sheet.getRange(`A1:A${lastRow}`).format.columnWidthPx = 220;
  sheet.getRange(`B1:B${lastRow}`).format.columnWidthPx = 720;
  sheet.getRange(`A1:B${lastRow}`).format.autofitRows();
  sheet.freezePanes.freezeRows(1);
}

async function buildVerdictWorkbook(blindRows, outputPath, ruleCards, { calibration = false } = {}) {
  const workbook = Workbook.create();
  const instructions = workbook.worksheets.add("标注说明");
  styleInstructionSheet(
    instructions,
    instructionRows(ruleCards, {
      title: calibration ? "RegGround-AV 校准轮（不计入正式样本）" : "RegGround-AV 人工盲标 v1",
      calibration,
    }),
  );

  const annotation = workbook.worksheets.add("盲标表");
  annotation.showGridLines = false;
  const matrix = [BLIND_HEADERS, ...blindRows.map((row) => BLIND_HEADERS.map((header) => row[header]))];
  const lastRow = matrix.length;
  annotation.getRange(`A1:N${lastRow}`).values = matrix;
  annotation.getRange("A1:N1").format = {
    fill: "#1F4E78",
    font: { bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
  annotation.getRange(`A2:N${lastRow}`).format = {
    verticalAlignment: "top",
    wrapText: true,
  };
  annotation.getRange(`I2:N${lastRow}`).format.fill = "#FFF2CC";
  annotation.getRange(`A1:N${lastRow}`).format.borders = {
    insideHorizontal: { style: "thin", color: "#D9E2F3" },
    bottom: { style: "thin", color: "#9EADBA" },
  };
  const widths = [90, 110, 300, 360, 400, 270, 105, 390, 115, 205, 205, 145, 320, 240];
  widths.forEach((width, index) => {
    annotation.getRangeByIndexes(0, index, lastRow, 1).format.columnWidthPx = width;
  });
  annotation.getRange(`A2:N${lastRow}`).format.rowHeightPx = 108;
  annotation.getRange("A1:N1").format.rowHeightPx = 74;
  annotation.getRange(`I2:I${lastRow}`).dataValidation = {
    rule: { type: "list", values: ["cleared", "vetoed", "uncertain"] },
  };
  annotation.getRange(`L2:L${lastRow}`).dataValidation = {
    rule: { type: "whole", operator: "between", formula1: 1, formula2: 5 },
  };
  annotation.tables.add(`A1:N${lastRow}`, true, "BlindAnnotationTable");
  annotation.freezePanes.freezeRows(1);
  annotation.freezePanes.freezeColumns(2);

  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);
  return workbook;
}

async function buildPreferenceWorkbook(groups, outputPath, ruleCards) {
  const workbook = Workbook.create();
  const instructionSheet = workbook.worksheets.add("标注说明");
  styleInstructionSheet(instructionSheet, instructionRows(ruleCards, {
    title: "RegGround-AV 场景级偏好盲标 v1",
    preference: true,
  }));

  const sheet = workbook.worksheets.add("场景偏好表");
  sheet.showGridLines = false;
  const maxCandidates = Math.max(...groups.map((group) => group.candidates.length));
  const headers = preferenceHeaders(maxCandidates);
  const matrix = [headers, ...groups.map((group) => toPreferenceRow(group, headers))];
  const lastRow = matrix.length;
  const lastColumn = columnName(headers.length);
  sheet.getRange(`A1:${lastColumn}${lastRow}`).values = matrix;
  sheet.getRange(`A1:${lastColumn}1`).format = {
    fill: "#1F4E78",
    font: { bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
  sheet.getRange(`A2:${lastColumn}${lastRow}`).format = { verticalAlignment: "top", wrapText: true };
  sheet.getRange(`A1:${lastColumn}${lastRow}`).format.borders = {
    insideHorizontal: { style: "thin", color: "#D9E2F3" },
    bottom: { style: "thin", color: "#9EADBA" },
  };
  const widths = [95, 110, 300, 360, 400, 270];
  for (let index = 0; index < maxCandidates; index += 1) widths.push(105, 390);
  widths.push(170, 260, 145, 320, 240);
  widths.forEach((width, index) => {
    sheet.getRangeByIndexes(0, index, lastRow, 1).format.columnWidthPx = width;
  });
  sheet.getRange(`A2:${lastColumn}${lastRow}`).format.rowHeightPx = 150;
  sheet.getRange(`A1:${lastColumn}1`).format.rowHeightPx = 74;
  const confidenceColumn = headers.indexOf("human_confidence");
  sheet.getRangeByIndexes(1, confidenceColumn, groups.length, 1).dataValidation = {
    rule: { type: "whole", operator: "between", formula1: 1, formula2: 5 },
  };
  sheet.tables.add(`A1:${lastColumn}${lastRow}`, true, "PreferenceAnnotationTable");
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(2);

  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);
  return workbook;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const inputPath = path.resolve(process.cwd(), options.input);
  const outputRoot = path.resolve(process.cwd(), options.outputRoot);
  const ruleGraphPath = path.resolve(process.cwd(), options.ruleGraph);
  const ruleCards = await loadRuleCards(ruleGraphPath);
  const resultsDir = path.join(outputRoot, "results");
  const privateDir = path.join(outputRoot, "private");
  await fs.mkdir(resultsDir, { recursive: true });
  await fs.mkdir(privateDir, { recursive: true });

  const inputText = await fs.readFile(inputPath, "utf8");
  const records = inputText
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line))
    .filter((item) => item.type === "record" && item.data?.ok && !item.data.skipped_by_admission)
    .map((item) => item.data);
  const candidates = buildCandidates(records);
  const profile = quotaProfile(candidates);
  const initialCore = selectCore(candidates, options.seed, profile.llmQuotas);
  const stress = selectStress(candidates, initialCore, options.seed);
  const core = enforceBorderlineVerdictShare(
    candidates,
    initialCore,
    stress,
    options.seed,
    profile.llmQuotas,
  );
  const samples = assignAuditIds([...core, ...stress], options.seed);
  const blindRows = samples.map(toBlind);
  const sidecarRows = samples.map(toSidecar);
  const preferenceGroups = selectPreferenceScenes(candidates, core, options.seed);
  const preferenceHeadersList = preferenceHeaders(
    Math.max(...preferenceGroups.map((group) => group.candidates.length)),
  );
  const preferenceRows = preferenceGroups.map((group) => toPreferenceObject(group, preferenceHeadersList));
  const calibrationSamples = selectCalibrationSamples(candidates, core, options.seed);
  const calibrationRows = calibrationSamples.map(toBlind);
  verify(samples, blindRows, profile.llmQuotas);
  verifyPreferenceGroups(preferenceGroups);

  const blindPath = path.join(resultsDir, "blind_samples.jsonl");
  const sidecarPath = path.join(privateDir, "hidden_sidecar.jsonl");
  const workbookPath = path.join(resultsDir, "blind_annotation_template.xlsx");
  const preferencePath = path.join(resultsDir, "preference_blind_samples.jsonl");
  const preferenceSidecarPath = path.join(privateDir, "preference_sidecar.jsonl");
  const preferenceWorkbookPath = path.join(resultsDir, "preference_annotation_template.xlsx");
  const calibrationPath = path.join(resultsDir, "calibration_blind_samples.jsonl");
  const calibrationSidecarPath = path.join(privateDir, "calibration_sidecar.jsonl");
  const calibrationWorkbookPath = path.join(resultsDir, "calibration_annotation_template.xlsx");
  await fs.writeFile(blindPath, `${blindRows.map((row) => JSON.stringify(row)).join("\n")}\n`);
  await fs.writeFile(sidecarPath, `${sidecarRows.map((row) => JSON.stringify(row)).join("\n")}\n`);
  await fs.writeFile(preferencePath, `${preferenceRows.map((row) => JSON.stringify(row)).join("\n")}\n`);
  await fs.writeFile(
    preferenceSidecarPath,
    `${preferenceGroups.map((group) => JSON.stringify({
      preference_id: group.preferenceId,
      scene_id: group.sceneId,
      frame_token: group.frameToken,
      candidate_ids: group.candidates.map((candidate) => candidate.trajectoryId),
      scoreable: group.scoreable,
    })).join("\n")}\n`,
  );
  await fs.writeFile(calibrationPath, `${calibrationRows.map((row) => JSON.stringify(row)).join("\n")}\n`);
  await fs.writeFile(
    calibrationSidecarPath,
    `${calibrationSamples.map(toCalibrationSidecar).map((row) => JSON.stringify(row)).join("\n")}\n`,
  );
  const workbook = await buildVerdictWorkbook(blindRows, workbookPath, ruleCards);
  const preferenceWorkbook = await buildPreferenceWorkbook(preferenceGroups, preferenceWorkbookPath, ruleCards);
  const calibrationWorkbook = await buildVerdictWorkbook(
    calibrationRows,
    calibrationWorkbookPath,
    ruleCards,
    { calibration: true },
  );

  const manifest = {
    protocol_version: "human_annotation_v1",
    seed: options.seed,
    reference_commit: options.referenceCommit,
    source: path.relative(outputRoot, inputPath),
    source_sha256: sha256Text(inputText),
    rule_graph: path.relative(outputRoot, ruleGraphPath),
    rule_graph_sha256: await sha256File(ruleGraphPath),
    rule_cards: ruleCards,
    source_records: records.length,
    source_candidates: candidates.length,
    sampling_profile: profile.name,
    core_llm_quotas: profile.llmQuotas,
    sample_counts: {
      total: samples.length,
      core: core.length,
      stress: stress.length,
      core_unique_scenes: new Set(core.map((sample) => sample.sceneId)).size,
    },
    core_strata: countBy(core, (sample) => sample.stratum),
    stress_strata: countBy(stress, (sample) => sample.stratum),
    sampled_system_statuses: countBy(samples, (sample) => sample.systemVerdict),
    sampled_sources: countBy(samples, (sample) => sample.source),
    sampled_sampling_bands: countBy(
      samples,
      (sample) => isBorderlineCandidate(sample) ? "borderline_medium" : "standard",
    ),
    borderline_sampling: {
      enabled: candidates.some(isBorderlineCandidate),
      reasons: BORDERLINE_VERDICT_SAMPLING.reasons,
      target_ratio: BORDERLINE_VERDICT_SAMPLING.targetRatio,
      min_ratio: BORDERLINE_VERDICT_SAMPLING.minRatio,
      max_ratio: BORDERLINE_VERDICT_SAMPLING.maxRatio,
      pool_count: candidates.filter(isBorderlineCandidate).length,
      sampled_count: samples.filter(isBorderlineCandidate).length,
      sampled_ratio: samples.filter(isBorderlineCandidate).length / samples.length,
    },
    forced_stress_case: FORCED_STRESS_CASE,
    outputs: {
      blind_samples: {
        path: path.relative(outputRoot, blindPath),
        sha256: await sha256File(blindPath),
      },
      hidden_sidecar: {
        path: path.relative(outputRoot, sidecarPath),
        sha256: await sha256File(sidecarPath),
      },
      workbook: {
        path: path.relative(outputRoot, workbookPath),
        sha256: await sha256File(workbookPath),
      },
      preference_blind_samples: {
        path: path.relative(outputRoot, preferencePath),
        sha256: await sha256File(preferencePath),
      },
      preference_sidecar: {
        path: path.relative(outputRoot, preferenceSidecarPath),
        sha256: await sha256File(preferenceSidecarPath),
      },
      preference_workbook: {
        path: path.relative(outputRoot, preferenceWorkbookPath),
        sha256: await sha256File(preferenceWorkbookPath),
      },
      calibration_blind_samples: {
        path: path.relative(outputRoot, calibrationPath),
        sha256: await sha256File(calibrationPath),
      },
      calibration_sidecar: {
        path: path.relative(outputRoot, calibrationSidecarPath),
        sha256: await sha256File(calibrationSidecarPath),
      },
      calibration_workbook: {
        path: path.relative(outputRoot, calibrationWorkbookPath),
        sha256: await sha256File(calibrationWorkbookPath),
      },
    },
    preference_annotation: {
      scene_count: preferenceGroups.length,
      scoreable_scene_count: preferenceGroups.filter((group) => group.scoreable).length,
      candidate_count_by_scene: countBy(preferenceGroups, (group) => String(group.candidates.length)),
    },
    calibration: {
      row_count: calibrationSamples.length,
      categories: countBy(calibrationSamples, (sample) => sample.calibrationKind),
    },
    interpretation: {
      core: "用于代表性人工一致性估计；按来源分层报告。",
      stress: "用于失败分析；不得与 core 合并估计总体准确率。",
    },
  };
  const manifestPath = path.join(resultsDir, "sample_manifest.json");
  await fs.writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  const artifactHashesPath = path.join(resultsDir, "artifact_hashes.json");
  await fs.writeFile(
    artifactHashesPath,
    `${JSON.stringify({
      protocol_version: "human_annotation_v1_hash_anchor",
      reference_commit: options.referenceCommit,
      artifacts: {
        blind_samples: await sha256File(blindPath),
        sample_manifest: await sha256File(manifestPath),
        blind_annotation_template: await sha256File(workbookPath),
        preference_annotation_template: await sha256File(preferenceWorkbookPath),
        calibration_annotation_template: await sha256File(calibrationWorkbookPath),
      },
    }, null, 2)}\n`,
  );

  const inspectFormulaErrors = (workbookToInspect, label) => workbookToInspect.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 50 },
    summary: `${label} formula error scan`,
  });
  const [formalErrors, preferenceErrors, calibrationErrors] = await Promise.all([
    inspectFormulaErrors(workbook, "formal verdict"),
    inspectFormulaErrors(preferenceWorkbook, "preference"),
    inspectFormulaErrors(calibrationWorkbook, "calibration"),
  ]);
  const instructionsPreview = await workbook.render({
    sheetName: "标注说明",
    range: "A1:B25",
    scale: 1,
    format: "png",
  });
  const annotationPreview = await workbook.render({
    sheetName: "盲标表",
    range: "A1:N8",
    scale: 0.8,
    format: "png",
  });
  const preferenceInstructionsPreview = await preferenceWorkbook.render({
    sheetName: "标注说明",
    range: "A1:B25",
    scale: 1,
    format: "png",
  });
  const preferenceAnnotationPreview = await preferenceWorkbook.render({
    sheetName: "场景偏好表",
    range: `A1:${columnName(preferenceHeadersList.length)}5`,
    scale: 0.7,
    format: "png",
  });
  const calibrationInstructionsPreview = await calibrationWorkbook.render({
    sheetName: "标注说明",
    range: "A1:B25",
    scale: 1,
    format: "png",
  });
  const calibrationAnnotationPreview = await calibrationWorkbook.render({
    sheetName: "盲标表",
    range: "A1:N8",
    scale: 0.8,
    format: "png",
  });
  await fs.writeFile(
    path.join(outputRoot, "qa_preview_instructions.png"),
    new Uint8Array(await instructionsPreview.arrayBuffer()),
  );
  await fs.writeFile(
    path.join(outputRoot, "qa_preview_annotation.png"),
    new Uint8Array(await annotationPreview.arrayBuffer()),
  );
  await fs.writeFile(
    path.join(outputRoot, "qa_preview_preference_instructions.png"),
    new Uint8Array(await preferenceInstructionsPreview.arrayBuffer()),
  );
  await fs.writeFile(
    path.join(outputRoot, "qa_preview_preference_annotation.png"),
    new Uint8Array(await preferenceAnnotationPreview.arrayBuffer()),
  );
  await fs.writeFile(
    path.join(outputRoot, "qa_preview_calibration_instructions.png"),
    new Uint8Array(await calibrationInstructionsPreview.arrayBuffer()),
  );
  await fs.writeFile(
    path.join(outputRoot, "qa_preview_calibration_annotation.png"),
    new Uint8Array(await calibrationAnnotationPreview.arrayBuffer()),
  );
  await fs.rm(`${workbookPath}.inspect.ndjson`, { force: true });
  await fs.rm(`${preferenceWorkbookPath}.inspect.ndjson`, { force: true });
  await fs.rm(`${calibrationWorkbookPath}.inspect.ndjson`, { force: true });

  process.stdout.write(
    `${JSON.stringify(
      {
        manifest,
        artifact_hashes: JSON.parse(await fs.readFile(artifactHashesPath, "utf8")),
        formula_errors: {
          formal: formalErrors.ndjson,
          preference: preferenceErrors.ndjson,
          calibration: calibrationErrors.ndjson,
        },
      },
      null,
      2,
    )}\n`,
  );
}

await main();
