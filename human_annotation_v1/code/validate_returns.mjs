import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";

import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

import { loadRuleCards, validateVerdictRow } from "./annotation_schema.mjs";

const DEFAULTS = {
  input: "results/rater_a_completed.xlsx",
  template: "results/blind_annotation_template.xlsx",
  ruleGraph: "../data/layer2/rule_graph.yaml",
};

const REQUIRED_HEADERS = [
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
    else if (key === "--template") options.template = value;
    else if (key === "--rule-graph") options.ruleGraph = value;
    else if (key === "--output") options.output = value;
    else throw new Error(`未知参数: ${key}`);
    index += 1;
  }
  return options;
}

function text(value) {
  return String(value ?? "").trim();
}

async function readAnnotationRows(filePath) {
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(filePath));
  const sheet = workbook.worksheets.getItem("盲标表");
  const values = sheet.getUsedRange().values;
  return values.map((row) => row.map(text));
}

function toRow(values) {
  return Object.fromEntries(REQUIRED_HEADERS.map((header, index) => [header, values[index] ?? ""]));
}

function compareTemplate(templateValues, returnedValues) {
  const issues = [];
  const expectedHeaders = templateValues[0] ?? [];
  if (expectedHeaders.join("|") !== REQUIRED_HEADERS.join("|")) {
    issues.push({ row: 1, audit_id: "", messages: ["模板表头与协议不一致"] });
  }
  if (returnedValues.length !== templateValues.length) {
    issues.push({
      row: 0,
      audit_id: "",
      messages: [`行数必须为 ${templateValues.length - 1} 条，实际为 ${returnedValues.length - 1}`],
    });
  }
  const expectedIds = templateValues.slice(1).map((row) => text(row[0]));
  const returnedIds = returnedValues.slice(1).map((row) => text(row[0]));
  if (expectedIds.length !== returnedIds.length || expectedIds.some((id, index) => id !== returnedIds[index])) {
    issues.push({ row: 0, audit_id: "", messages: ["audit_id 集合或顺序与模板不一致"] });
  }
  return issues;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const inputPath = path.resolve(process.cwd(), options.input);
  const templatePath = path.resolve(process.cwd(), options.template);
  const ruleGraphPath = path.resolve(process.cwd(), options.ruleGraph);
  const outputPath = path.resolve(
    process.cwd(),
    options.output ?? `results/return_validation_${path.basename(inputPath, ".xlsx")}.json`,
  );
  const ruleCards = await loadRuleCards(ruleGraphPath);
  const templateValues = await readAnnotationRows(templatePath);
  const returnedValues = await readAnnotationRows(inputPath);
  const issues = compareTemplate(templateValues, returnedValues);
  const normalizedRows = [];
  for (let index = 1; index < returnedValues.length; index += 1) {
    const row = toRow(returnedValues[index]);
    const result = validateVerdictRow(row, ruleCards);
    if (result.issues.length > 0) {
      issues.push({ row: index + 1, audit_id: row.audit_id, messages: result.issues });
    }
    normalizedRows.push({
      row: index + 1,
      audit_id: row.audit_id,
      human_applicable_rule_ids: result.applicable.canonical,
      human_violated_rule_ids: result.violated.canonical,
    });
  }
  const report = {
    protocol_version: "human_annotation_v1_return_validation",
    input: path.relative(process.cwd(), inputPath),
    template: path.relative(process.cwd(), templatePath),
    rule_graph: path.relative(process.cwd(), ruleGraphPath),
    expected_rows: templateValues.length - 1,
    returned_rows: returnedValues.length - 1,
    ok: issues.length === 0,
    issue_count: issues.length,
    issues,
    normalized_rows: normalizedRows,
  };
  await fs.writeFile(
    outputPath,
    `${JSON.stringify(report, null, 2)}\n`,
    "utf8",
  );
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  if (!report.ok) process.exitCode = 1;
}

await main();
