import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const [rowsPath, outputPath, previewPath] = process.argv.slice(2);
if (!rowsPath || !outputPath || !previewPath) {
  throw new Error("usage: build_citation_sheet.mjs ROWS_JSON OUTPUT_XLSX PREVIEW_PNG");
}

const rows = JSON.parse(await fs.readFile(rowsPath, "utf8"));
const headers = [
  "citation_id",
  "condition",
  "scene_id",
  "frame_token",
  "attempt",
  "owner",
  "original_text",
  "law_name",
  "article_number",
  "normalized_key",
  "automated_validity",
  "existence_rating",
  "content_fidelity_rating",
  "applicability_rating",
  "notes",
];
const matrix = [headers, ...rows.map((row) => headers.map((header) => row[header] ?? ""))];

const workbook = Workbook.create();
const sheet = workbook.worksheets.add("引用评级");
sheet.showGridLines = false;
sheet.getRangeByIndexes(0, 0, matrix.length, headers.length).values = matrix;
sheet.freezePanes.freezeRows(1);
sheet.freezePanes.freezeColumns(2);

const header = sheet.getRange(`A1:O1`);
header.format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF" },
  rowHeight: 30,
  wrapText: true,
  verticalAlignment: "center",
};
const dataEnd = Math.max(2, matrix.length);
const dataRange = sheet.getRange(`A2:O${dataEnd}`);
dataRange.format = {
  font: { color: "#1F2937" },
  verticalAlignment: "top",
  borders: { preset: "inside", style: "thin", color: "#D9E2F3" },
};
sheet.getRange(`G2:G${dataEnd}`).format.wrapText = true;
sheet.getRange(`O2:O${dataEnd}`).format.wrapText = true;
sheet.getRange(`I2:I${dataEnd}`).format.numberFormat = "0";
sheet.getRange(`L2:N${dataEnd}`).dataValidation = {
  rule: { type: "list", values: ["待评级", "是", "否", "不确定"] },
};
sheet.getRange(`L2:N${dataEnd}`).conditionalFormats.add("containsText", {
  text: "待评级",
  format: { fill: "#FFF2CC", font: { color: "#7F6000" } },
});
sheet.getRange(`L2:N${dataEnd}`).conditionalFormats.add("containsText", {
  text: "否",
  format: { fill: "#FCE4D6", font: { color: "#9C0006" } },
});
sheet.getRange("A:A").format.columnWidth = 38;
sheet.getRange("B:B").format.columnWidth = 16;
sheet.getRange("C:F").format.columnWidth = 24;
sheet.getRange("G:G").format.columnWidth = 34;
sheet.getRange("H:J").format.columnWidth = 20;
sheet.getRange("K:N").format.columnWidth = 19;
sheet.getRange("O:O").format.columnWidth = 26;
if (matrix.length > 1) {
  const table = sheet.tables.add(`A1:O${matrix.length}`, true, "CitationGradingTable");
  table.style = "TableStyleMedium2";
  table.showFilterButton = true;
}

const inspection = await workbook.inspect({
  kind: "table",
  range: `引用评级!A1:O${Math.min(dataEnd, 8)}`,
  include: "values,formulas",
  tableMaxRows: 8,
  tableMaxCols: 15,
});
if (!inspection.ndjson.includes("citation_id")) {
  throw new Error("workbook verification failed: header not found");
}
const formulaErrors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 20 },
  summary: "formula error scan",
});
if (/\"matches\"\s*:\s*\[[^\]]/.test(formulaErrors.ndjson)) {
  throw new Error("workbook verification failed: formula error found");
}
const preview = await workbook.render({
  sheetName: "引用评级",
  range: `A1:O${Math.min(dataEnd, 12)}`,
  scale: 1,
  format: "png",
});
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
