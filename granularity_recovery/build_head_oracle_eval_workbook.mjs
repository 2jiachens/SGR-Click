import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const repoRoot = process.cwd();
const packageDir = path.join(repoRoot, "outputs", "packages", "head_oracle_eval_validation_20260705");
const summaryPath = path.join(packageDir, "final_summary.json");
const samplesCsvPath = path.join(packageDir, "sample_metrics.csv");
const skippedCsvPath = path.join(packageDir, "skipped_samples.csv");
const outputXlsxPath = path.join(packageDir, "head_oracle_eval_validation.xlsx");
const previewPath = path.join(packageDir, "head_oracle_eval_validation_preview.png");

const summary = JSON.parse(await fs.readFile(summaryPath, "utf8"));
const samplesCsv = await fs.readFile(samplesCsvPath, "utf8");
const skippedCsv = await fs.readFile(skippedCsvPath, "utf8");

function parseCsv(text) {
  const clean = text.replace(/^\uFEFF/, "").trim();
  if (!clean) return [];
  return clean.split(/\r?\n/).map((line) => line.split(","));
}

function colName(index) {
  let n = index + 1;
  let out = "";
  while (n > 0) {
    const rem = (n - 1) % 26;
    out = String.fromCharCode(65 + rem) + out;
    n = Math.floor((n - 1) / 26);
  }
  return out;
}

function writeMatrix(sheet, matrix) {
  if (!matrix.length || !matrix[0].length) return;
  const rows = matrix.length;
  const cols = matrix[0].length;
  const range = `${colName(0)}1:${colName(cols - 1)}${rows}`;
  sheet.getRange(range).values = matrix;
}

const workbook = Workbook.create();

const summarySheet = workbook.worksheets.add("Summary");
summarySheet.showGridLines = false;
summarySheet.freezePanes.freezeRows(3);

summarySheet.getRange("A1:F1").merge();
summarySheet.getRange("A1").values = [["Oracle Head Validation Summary"]];
summarySheet.getRange("A1:F1").format.font = { bold: true, size: 16, color: "#FFFFFF" };
summarySheet.getRange("A1:F1").format.fill = { color: "#1F4E78" };
summarySheet.getRange("A1:F1").format.horizontalAlignment = "center";
summarySheet.getRange("A1:F1").format.verticalAlignment = "center";
summarySheet.getRange("A1:F1").format.rowHeight = 26;

summarySheet.getRange("A3:B10").values = [
  ["Metric", "Value"],
  ["Evaluated Samples", summary.evaluated_samples],
  ["Skipped Samples", summary.skipped_samples],
  ["Target Total Samples", summary.target_total_samples],
  ["NoC@85", summary.noc85_mean],
  ["IoU@1", summary.iou1_mean],
  ["Failures@20", summary.failures_at_20],
  ["Package Folder", summary.package_dir],
];
summarySheet.getRange("A3:B3").format.font = { bold: true, color: "#FFFFFF" };
summarySheet.getRange("A3:B3").format.fill = { color: "#4F81BD" };
summarySheet.getRange("A3:B10").format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
summarySheet.getRange("B4:B9").format.horizontalAlignment = "right";
summarySheet.getRange("B7:B8").setNumberFormat("0.0000");
summarySheet.getRange("A12:B16").values = [
  ["Source Folder", "Role"],
  [path.join(repoRoot, "outputs", "head_oracle_eval"), "Primary oracle evaluation results"],
  [path.join(repoRoot, "outputs", "head_oracle_eval_repair"), "Missing-sample repair results"],
  [samplesCsvPath, "Merged per-sample metrics CSV"],
  [skippedCsvPath, "Skipped sample list CSV"],
];
summarySheet.getRange("A12:B12").format.font = { bold: true, color: "#FFFFFF" };
summarySheet.getRange("A12:B12").format.fill = { color: "#8064A2" };
summarySheet.getRange("A12:B16").format.borders = { preset: "all", style: "thin", color: "#E4DFEC" };
summarySheet.getRange("A:B").format.wrapText = true;
summarySheet.getRange("A:B").format.autofitColumns();
summarySheet.getRange("A:B").format.columnWidthPx = 260;

const samplesSheet = workbook.worksheets.add("Sample Metrics");
writeMatrix(samplesSheet, parseCsv(samplesCsv));
samplesSheet.showGridLines = false;
samplesSheet.freezePanes.freezeRows(1);
samplesSheet.getRange("A1:L1").format.font = { bold: true, color: "#FFFFFF" };
samplesSheet.getRange("A1:L1").format.fill = { color: "#4F81BD" };
samplesSheet.getRange("A1:L2000").format.borders = { preset: "all", style: "thin", color: "#E6EEF8" };
samplesSheet.getRange("F2:J2000").format.horizontalAlignment = "right";
samplesSheet.getRange("G2:H2000").setNumberFormat("0.0000");
samplesSheet.getRange("K2:K2000").setNumberFormat("0.00");
samplesSheet.getRange("A:L").format.autofitColumns();

const skippedSheet = workbook.worksheets.add("Skipped");
writeMatrix(skippedSheet, parseCsv(skippedCsv));
skippedSheet.showGridLines = false;
skippedSheet.freezePanes.freezeRows(1);
skippedSheet.getRange("A1:A2").format.borders = { preset: "all", style: "thin", color: "#F4CCCC" };
skippedSheet.getRange("A1").format.font = { bold: true, color: "#FFFFFF" };
skippedSheet.getRange("A1").format.fill = { color: "#C0504D" };
skippedSheet.getRange("A:A").format.autofitColumns();

const inspect = await workbook.inspect({
  kind: "table",
  range: "A1:L8",
  sheetId: samplesSheet.id,
  include: "values,formulas",
  tableMaxRows: 8,
  tableMaxCols: 12,
});
console.log(inspect.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 50 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const preview = await workbook.render({ sheetName: "Summary", range: "A1:B16", scale: 2, format: "png" });
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputXlsxPath);

console.log(JSON.stringify({ outputXlsxPath, previewPath }));
