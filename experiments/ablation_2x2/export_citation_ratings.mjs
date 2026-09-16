import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const [workbookPath, outputPath] = process.argv.slice(2);
if (!workbookPath || !outputPath) {
  throw new Error("usage: export_citation_ratings.mjs INPUT_XLSX OUTPUT_JSON");
}

const input = await FileBlob.load(workbookPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const sheet = workbook.worksheets.getItem("引用评级");
const values = sheet.getUsedRange(true).values;
if (!Array.isArray(values) || values.length < 2) {
  throw new Error("引用评级 sheet is empty");
}
const headers = values[0].map((value) => String(value));
const rows = values.slice(1).map((valuesRow) =>
  Object.fromEntries(headers.map((header, index) => [header, valuesRow[index] ?? ""])),
);
await fs.writeFile(outputPath, `${JSON.stringify(rows, null, 2)}\n`, "utf8");
