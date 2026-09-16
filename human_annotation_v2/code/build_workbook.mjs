#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

function parseArgs(argv) {
  const args = {};
  for (let index = 0; index < argv.length; index += 2) {
    args[argv[index]] = argv[index + 1];
  }
  for (const required of ["--spec", "--output", "--preview-dir"]) {
    if (!args[required]) throw new Error(`缺少参数 ${required}`);
  }
  return args;
}

function columnName(index) {
  let name = "";
  for (let value = index + 1; value > 0; value = Math.floor((value - 1) / 26)) {
    name = String.fromCharCode(65 + ((value - 1) % 26)) + name;
  }
  return name;
}

function safeFileName(name) {
  return name.replace(/[^\p{Letter}\p{Number}_.-]+/gu, "_");
}

function applyValidation(sheet, header, rowCount) {
  if (rowCount === 0) return;
  const rangeFor = (name) => {
    const index = header.indexOf(name);
    if (index < 0) return null;
    const column = columnName(index);
    return sheet.getRange(`${column}2:${column}${rowCount + 1}`);
  };

  const verdict = rangeFor("human_verdict");
  if (verdict) {
    verdict.dataValidation = {
      rule: { type: "list", values: ["cleared", "vetoed", "uncertain"] },
    };
  }
  const confidence = rangeFor("human_confidence");
  if (confidence) {
    confidence.dataValidation = {
      rule: { type: "whole", operator: "between", formula1: 1, formula2: 5 },
    };
  }
  const best = rangeFor("human_best_trajectory");
  if (best) {
    const candidates = header.filter((name) => name.startsWith("cand_"));
    best.dataValidation = {
      rule: { type: "list", values: ["NONE", ...candidates] },
    };
  }
}

function populateSheet(workbook, spec) {
  const sheet = workbook.worksheets.add(spec.name);
  sheet.showGridLines = false;
  const values = [spec.header, ...spec.rows];
  const lastColumn = columnName(spec.header.length - 1);
  const used = sheet.getRange(`A1:${lastColumn}${values.length}`);
  used.values = values;
  used.format = {
    font: { name: "Aptos", size: 10, color: "#1F2937" },
    verticalAlignment: "top",
    wrapText: true,
  };
  sheet.getRange(`A1:${lastColumn}1`).format = {
    fill: "#17365D",
    font: { name: "Aptos", size: 10, bold: true, color: "#FFFFFF" },
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "outside", style: "thin", color: "#9FBAD0" },
  };
  sheet.getRange(`A1:${lastColumn}1`).format.rowHeight = 32;

  if (spec.freeze_header) {
    sheet.freezePanes.freezeRows(1);
    if (spec.name === "标注") sheet.freezePanes.freezeColumns(1);
  }

  for (let index = 0; index < spec.header.length; index += 1) {
    const width = spec.column_widths[index] ?? 18;
    sheet.getRangeByIndexes(0, index, values.length, 1).format.columnWidth = Math.min(width, 58);
  }

  if (spec.rows.length > 0) {
    sheet.getRange(`A2:${lastColumn}${values.length}`).format.rowHeight =
      spec.name === "标注" ? 108 : 30;
  }

  for (const column of spec.mono_columns ?? []) {
    const index = spec.header.indexOf(column);
    if (index < 0 || spec.rows.length === 0) continue;
    const letter = columnName(index);
    sheet.getRange(`${letter}2:${letter}${values.length}`).format.font = {
      name: "Aptos Mono",
      size: 9,
      color: "#243447",
    };
  }

  for (const [rowIndexText, style] of Object.entries(spec.row_styles ?? {})) {
    if (Number(style) !== 3) continue;
    const excelRow = Number(rowIndexText) + 2;
    sheet.getRange(`A${excelRow}:${lastColumn}${excelRow}`).format = {
      fill: "#FFF4CC",
      font: { name: "Aptos", size: 10, italic: true, color: "#7A4E00" },
      verticalAlignment: "top",
      wrapText: true,
    };
  }

  const editableIndexes = spec.header
    .map((name, index) => (name.startsWith("human_") || name === "notes" ? index : -1))
    .filter((index) => index >= 0);
  for (const index of editableIndexes) {
    if (spec.rows.length === 0) continue;
    const letter = columnName(index);
    sheet.getRange(`${letter}2:${letter}${values.length}`).format.fill = "#FFF8D6";
  }
  applyValidation(sheet, spec.header, spec.rows.length);
  return { sheet, lastColumn, rowCount: values.length };
}

const args = parseArgs(process.argv.slice(2));
const spec = JSON.parse(await fs.readFile(args["--spec"], "utf8"));
const workbook = Workbook.create();
const populated = spec.sheets.map((sheetSpec) => ({
  spec: sheetSpec,
  ...populateSheet(workbook, sheetSpec),
}));

await fs.mkdir(path.dirname(args["--output"]), { recursive: true });
await fs.mkdir(args["--preview-dir"], { recursive: true });

const qa = [];
for (const item of populated) {
  const previewRows = item.spec.name === "标注" ? Math.min(item.rowCount, 6) : item.rowCount;
  const range = `A1:${item.lastColumn}${previewRows}`;
  const inspection = await workbook.inspect({
    kind: "table",
    range: `${item.spec.name}!${range}`,
    tableMaxRows: Math.min(previewRows, 8),
    tableMaxCols: Math.min(item.spec.header.length, 14),
    tableMaxCellChars: 120,
    maxChars: 8000,
  });
  qa.push({ sheet: item.spec.name, range, inspection: inspection.ndjson });
  const preview = await workbook.render({
    sheetName: item.spec.name,
    range,
    scale: 0.8,
    format: "png",
  });
  const bytes = new Uint8Array(await preview.arrayBuffer());
  const stem = safeFileName(path.basename(args["--output"], ".xlsx"));
  await fs.writeFile(
    path.join(args["--preview-dir"], `${stem}_${safeFileName(item.spec.name)}.png`),
    bytes,
  );
}

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
qa.push({ formulaErrorScan: errors.ndjson });
const stem = safeFileName(path.basename(args["--output"], ".xlsx"));
await fs.writeFile(
  path.join(args["--preview-dir"], `${stem}_qa.json`),
  JSON.stringify(qa, null, 2),
  "utf8",
);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(args["--output"]);
