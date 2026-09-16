import fs from "node:fs/promises";

export const VERDICTS = new Set(["cleared", "vetoed", "uncertain"]);

function scalar(value) {
  const text = String(value ?? "").trim();
  if ((text.startsWith('"') && text.endsWith('"')) || (text.startsWith("'") && text.endsWith("'"))) {
    return text.slice(1, -1);
  }
  return text;
}

export function parseRuleCards(text) {
  const cards = [];
  let inRules = false;
  let current = null;
  const finish = () => {
    if (current) cards.push(current);
    current = null;
  };
  for (const line of text.split(/\r?\n/)) {
    if (/^rules:\s*$/.test(line)) {
      inRules = true;
      continue;
    }
    if (!inRules) continue;
    const idMatch = line.match(/^\s{2}-\s+id:\s*(\S+)\s*$/);
    if (idMatch) {
      finish();
      current = { ruleId: scalar(idMatch[1]), description: "", severity: "" };
      continue;
    }
    if (!current) continue;
    const fieldMatch = line.match(/^\s{4}(description|severity):\s*(.*?)\s*$/);
    if (fieldMatch) current[fieldMatch[1]] = scalar(fieldMatch[2]);
  }
  finish();
  if (cards.length === 0 || cards.some((card) => !card.ruleId || !card.description || !card.severity)) {
    throw new Error("规则图谱未解析出完整规则卡");
  }
  return cards;
}

export async function loadRuleCards(ruleGraphPath) {
  return parseRuleCards(await fs.readFile(ruleGraphPath, "utf8"));
}

export function normalizeRuleIds(value, ruleCards) {
  const raw = String(value ?? "").trim();
  if (!raw || raw.toUpperCase() === "NONE") {
    return { ids: [], canonical: "NONE", errors: [] };
  }
  const tokens = raw.replace(/[；，,、\n]+/g, ";").split(";").map((item) => item.trim()).filter(Boolean);
  const errors = [];
  if (tokens.some((token) => token.toUpperCase() === "NONE")) {
    errors.push("NONE 不能与其他 rule ID 混用");
  }
  const allowed = new Set(ruleCards.map((card) => card.ruleId));
  const ids = [];
  for (const token of tokens) {
    const match = token.match(/^R-([A-Z]+)-(\d+)$/i);
    const canonical = match
      ? `R-${match[1].toUpperCase()}-${match[2].padStart(2, "0")}`
      : token.toUpperCase();
    if (!allowed.has(canonical)) errors.push(`非法或未知 rule ID: ${token}`);
    if (!ids.includes(canonical)) ids.push(canonical);
    else errors.push(`重复 rule ID: ${canonical}`);
  }
  return { ids, canonical: ids.length ? ids.join(";") : "NONE", errors };
}

export function validateVerdictRow(row, ruleCards) {
  const issues = [];
  const verdict = String(row.human_verdict ?? "").trim().toLowerCase();
  if (!VERDICTS.has(verdict)) issues.push("human_verdict 必须是 cleared/vetoed/uncertain");

  const applicable = normalizeRuleIds(row.human_applicable_rule_ids, ruleCards);
  const violated = normalizeRuleIds(row.human_violated_rule_ids, ruleCards);
  issues.push(...applicable.errors.map((message) => `applicable: ${message}`));
  issues.push(...violated.errors.map((message) => `violated: ${message}`));
  if ((verdict === "cleared" || verdict === "uncertain") && violated.ids.length > 0) {
    issues.push(`${verdict} 的 human_violated_rule_ids 必须为 NONE`);
  }
  if (verdict === "vetoed" && violated.ids.length === 0) {
    issues.push("vetoed 至少需要一条 violated rule ID");
  }
  const hardIds = new Set(ruleCards.filter((card) => card.severity === "hard").map((card) => card.ruleId));
  if (verdict === "vetoed" && violated.ids.some((ruleId) => !hardIds.has(ruleId))) {
    issues.push("vetoed 的 violated rule ID 必须全部为 hard 规则");
  }

  const confidence = String(row.human_confidence ?? "").trim();
  if (!/^[1-5]$/.test(confidence)) issues.push("human_confidence 必须是 1–5 的整数");
  if (!String(row.human_reason ?? "").trim()) issues.push("human_reason 不能为空");
  return {
    verdict,
    applicable,
    violated,
    issues,
  };
}
