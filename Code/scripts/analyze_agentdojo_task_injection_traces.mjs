#!/usr/bin/env node

import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import {
  mkdir,
  readFile,
  readdir,
  stat,
  writeFile,
} from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const ATTACK_ORDER = [
  "direct",
  "ignore_previous",
  "injecagent",
  "important_instructions",
  "tool_knowledge",
];
const SUITE_ORDER = ["workspace", "slack", "travel", "banking"];

function usage() {
  return `Usage:
  node Code/scripts/analyze_agentdojo_task_injection_traces.mjs <experiment-root|traces-dir> [options]

Options:
  --output DIR       Output directory (default: <experiment-root>/analysis)
  --verify-hashes    Recompute every raw_trace.json SHA-256 digest
  --help             Show this help

Outputs:
  trace_statistics.md       Human-readable summary
  trace_statistics.json     Machine-readable aggregate statistics
  group_statistics.csv      Overall/attack/suite/attack-suite statistics
  trace_records.csv         One normalized row per trace
`;
}

function parseArgs(argv) {
  let input = null;
  let output = null;
  let verifyHashes = false;
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--help" || arg === "-h") {
      console.log(usage());
      process.exit(0);
    }
    if (arg === "--verify-hashes") {
      verifyHashes = true;
      continue;
    }
    if (arg === "--output") {
      if (i + 1 >= argv.length) throw new Error("--output requires a directory");
      output = argv[i + 1];
      i += 1;
      continue;
    }
    if (arg.startsWith("-")) throw new Error(`Unknown option: ${arg}`);
    if (input !== null) throw new Error(`Unexpected positional argument: ${arg}`);
    input = arg;
  }
  if (input === null) throw new Error("An experiment root or traces directory is required");
  return { input, output, verifyHashes };
}

async function exists(target) {
  try {
    await stat(target);
    return true;
  } catch (error) {
    if (error?.code === "ENOENT") return false;
    throw error;
  }
}

async function resolveRoots(inputArg, outputArg) {
  const input = path.resolve(inputArg);
  if (!(await exists(input))) throw new Error(`Input does not exist: ${input}`);
  const inputStat = await stat(input);
  if (!inputStat.isDirectory()) throw new Error(`Input is not a directory: ${input}`);

  const nestedTraces = path.join(input, "traces");
  const isTracesDirectory = path.basename(input).toLowerCase() === "traces";
  const tracesRoot = isTracesDirectory ? input : nestedTraces;
  if (!(await exists(tracesRoot))) {
    throw new Error(`Could not find traces directory: ${tracesRoot}`);
  }
  const experimentRoot = isTracesDirectory ? path.dirname(input) : input;
  const outputRoot = path.resolve(outputArg ?? path.join(experimentRoot, "analysis"));
  return { experimentRoot, tracesRoot, outputRoot };
}

function readJson(text, filePath) {
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new Error(`Invalid JSON in ${filePath}: ${error.message}`);
  }
}

async function sha256File(filePath) {
  const digest = createHash("sha256");
  await new Promise((resolve, reject) => {
    const stream = createReadStream(filePath);
    stream.on("data", (chunk) => digest.update(chunk));
    stream.on("end", resolve);
    stream.on("error", reject);
  });
  return digest.digest("hex");
}

function asNumber(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function boolValue(value) {
  return typeof value === "boolean" ? value : null;
}

function countMessages(rawTrace) {
  const messages = Array.isArray(rawTrace?.trajectory?.messages)
    ? rawTrace.trajectory.messages
    : [];
  let assistantMessages = 0;
  let toolMessages = 0;
  let toolCalls = 0;
  let toolErrors = 0;
  for (const message of messages) {
    if (message?.role === "assistant") assistantMessages += 1;
    if (message?.role === "tool") {
      toolMessages += 1;
      if (message.error !== null && message.error !== undefined && message.error !== "") {
        toolErrors += 1;
      }
    }
    if (Array.isArray(message?.tool_calls)) toolCalls += message.tool_calls.length;
  }
  return {
    message_count: messages.length,
    assistant_message_count: assistantMessages,
    tool_message_count: toolMessages,
    tool_call_count: toolCalls,
    tool_error_count: toolErrors,
  };
}

function addCount(map, value) {
  const key = value === null || value === undefined || value === "" ? "<missing>" : String(value);
  map.set(key, (map.get(key) ?? 0) + 1);
}

function mapToSortedObject(map) {
  return Object.fromEntries([...map.entries()].sort(([a], [b]) => a.localeCompare(b)));
}

function preferredSort(values, preferred) {
  const rank = new Map(preferred.map((value, index) => [value, index]));
  return [...values].sort((a, b) => {
    const rankA = rank.has(a) ? rank.get(a) : preferred.length;
    const rankB = rank.has(b) ? rank.get(b) : preferred.length;
    return rankA - rankB || String(a).localeCompare(String(b));
  });
}

async function loadTraceDirectory(directory, verifyHashes, integrity) {
  const sampleDir = directory.name;
  const directoryPath = path.join(directory.parentPath, directory.name);
  const truthPath = path.join(directoryPath, "ground_truth.json");
  const rawPath = path.join(directoryPath, "raw_trace.json");
  const hashPath = path.join(directoryPath, "raw_trace.sha256");
  const missing = [];
  for (const [label, filePath] of [
    ["ground_truth.json", truthPath],
    ["raw_trace.json", rawPath],
    ["raw_trace.sha256", hashPath],
  ]) {
    if (!(await exists(filePath))) missing.push(label);
  }
  if (missing.length > 0) {
    integrity.errors.push({ sample_dir: sampleDir, error: `missing ${missing.join(", ")}` });
    return null;
  }

  let truth;
  let rawTrace;
  let truthText;
  let rawBytes;
  let sidecarText;
  try {
    [truthText, rawBytes, sidecarText] = await Promise.all([
      readFile(truthPath, "utf8"),
      readFile(rawPath),
      readFile(hashPath, "utf8"),
    ]);
    const zeroByteFiles = [];
    if (Buffer.byteLength(truthText, "utf8") === 0) zeroByteFiles.push("ground_truth.json");
    if (rawBytes.length === 0) zeroByteFiles.push("raw_trace.json");
    if (Buffer.byteLength(sidecarText, "utf8") === 0) zeroByteFiles.push("raw_trace.sha256");
    if (zeroByteFiles.length > 0) {
      for (const filename of zeroByteFiles) {
        integrity.zero_byte_files[filename] += 1;
      }
      integrity.errors.push({
        sample_dir: sampleDir,
        error: `zero-byte file(s): ${zeroByteFiles.join(", ")}`,
      });
      return null;
    }
    truth = readJson(truthText, truthPath);
    rawTrace = readJson(rawBytes.toString("utf8"), rawPath);
  } catch (error) {
    integrity.errors.push({ sample_dir: sampleDir, error: error.message });
    return null;
  }

  const sampleId = rawTrace.sample_id ?? truth.sample_id ?? sampleDir;
  if (sampleDir !== sampleId || truth.sample_id !== sampleId || rawTrace.sample_id !== sampleId) {
    integrity.errors.push({
      sample_dir: sampleDir,
      error: "sample_id mismatch between directory, ground truth, and raw trace",
      truth_sample_id: truth.sample_id ?? null,
      raw_sample_id: rawTrace.sample_id ?? null,
    });
  }

  const sidecarDigest = sidecarText.trim().split(/\s+/)[0] ?? "";
  const truthDigest = truth.trace_sha256 ?? "";
  if (!/^[a-f0-9]{64}$/i.test(sidecarDigest)) {
    integrity.errors.push({ sample_id: sampleId, error: "invalid raw_trace.sha256 sidecar" });
  }
  if (truthDigest !== sidecarDigest) {
    integrity.errors.push({
      sample_id: sampleId,
      error: "ground-truth trace_sha256 does not match sidecar",
      truth_sha256: truthDigest,
      sidecar_sha256: sidecarDigest,
    });
  }
  if (verifyHashes) {
    const actualDigest = await sha256File(rawPath);
    integrity.hashes_checked += 1;
    if (actualDigest !== sidecarDigest) {
      integrity.errors.push({
        sample_id: sampleId,
        error: "raw trace SHA-256 mismatch",
        actual_sha256: actualDigest,
        expected_sha256: sidecarDigest,
      });
    }
  }

  const task = rawTrace.task ?? {};
  const provenance = rawTrace.provenance ?? {};
  const messageStats = countMessages(rawTrace);
  const injections = task.injections && typeof task.injections === "object" ? task.injections : {};
  return {
    sample_id: sampleId,
    suite_name: rawTrace.suite_name ?? null,
    attack_type: task.attack_type ?? null,
    user_task_id: task.user_task_id ?? null,
    injection_task_id: task.injection_task_id ?? null,
    valid: boolValue(truth.valid),
    invalid_reason: truth.invalid_reason ?? null,
    payload_present: boolValue(truth.payload_present),
    hijacked: boolValue(truth.hijacked),
    utility_passed: boolValue(truth.utility_passed),
    raw_agentdojo_security: boolValue(truth.raw_agentdojo_security),
    duration_seconds: asNumber(provenance.duration_seconds),
    evaluation_timestamp: provenance.evaluation_timestamp ?? null,
    run_error: provenance.run_error ?? null,
    run_id: provenance.run_id ?? null,
    seed: provenance.seed ?? null,
    benchmark_version: rawTrace.benchmark_version ?? null,
    model_id: rawTrace?.target?.model_id ?? null,
    pipeline_name: rawTrace?.target?.pipeline_name ?? null,
    trace_sha256: sidecarDigest,
    raw_trace_bytes: rawBytes.length,
    injection_site_count: Object.keys(injections).length,
    ...messageStats,
  };
}

function safeRate(numerator, denominator) {
  return denominator === 0 ? null : numerator / denominator;
}

function mean(values) {
  const finite = values.filter((value) => typeof value === "number" && Number.isFinite(value));
  if (finite.length === 0) return null;
  return finite.reduce((sum, value) => sum + value, 0) / finite.length;
}

function percentile(values, probability) {
  const finite = values
    .filter((value) => typeof value === "number" && Number.isFinite(value))
    .sort((a, b) => a - b);
  if (finite.length === 0) return null;
  const index = Math.max(0, Math.ceil(probability * finite.length) - 1);
  return finite[index];
}

function aggregate(records) {
  const total = records.length;
  const validRecords = records.filter((record) => record.valid === true);
  const invalidRecords = records.filter((record) => record.valid === false);
  const missingValid = total - validRecords.length - invalidRecords.length;
  const payloadRecords = validRecords.filter((record) => record.payload_present === true);
  const hijackedRecords = validRecords.filter((record) => record.hijacked === true);
  const hijackedPayloadRecords = payloadRecords.filter((record) => record.hijacked === true);
  const utilityRecords = validRecords.filter((record) => record.utility_passed === true);
  const resistedRecords = validRecords.filter((record) => record.hijacked === false);
  const rawSecurityTrue = records.filter((record) => record.raw_agentdojo_security === true).length;
  const rawSecurityFalse = records.filter((record) => record.raw_agentdojo_security === false).length;

  const counts = {
    total,
    valid: validRecords.length,
    invalid: invalidRecords.length,
    missing_valid_label: missingValid,
    payload_present_valid: payloadRecords.length,
    payload_absent_valid: validRecords.filter((record) => record.payload_present === false).length,
    hijacked_valid: hijackedRecords.length,
    resisted_valid: resistedRecords.length,
    missing_hijacked_label_valid: validRecords.filter((record) => record.hijacked === null).length,
    hijacked_payload_valid: hijackedPayloadRecords.length,
    utility_passed_valid: utilityRecords.length,
    utility_failed_valid: validRecords.filter((record) => record.utility_passed === false).length,
    missing_utility_label_valid: validRecords.filter((record) => record.utility_passed === null).length,
    hijacked_and_utility_passed: validRecords.filter(
      (record) => record.hijacked === true && record.utility_passed === true,
    ).length,
    hijacked_and_utility_failed: validRecords.filter(
      (record) => record.hijacked === true && record.utility_passed === false,
    ).length,
    resisted_and_utility_passed: validRecords.filter(
      (record) => record.hijacked === false && record.utility_passed === true,
    ).length,
    resisted_and_utility_failed: validRecords.filter(
      (record) => record.hijacked === false && record.utility_passed === false,
    ).length,
    raw_agentdojo_security_true: rawSecurityTrue,
    raw_agentdojo_security_false: rawSecurityFalse,
  };
  const rates = {
    valid_rate: safeRate(counts.valid, total),
    payload_present_rate_among_valid: safeRate(counts.payload_present_valid, counts.valid),
    hijack_rate_among_valid: safeRate(counts.hijacked_valid, counts.valid),
    ASR: safeRate(counts.hijacked_payload_valid, counts.payload_present_valid),
    utility_pass_rate_among_valid: safeRate(counts.utility_passed_valid, counts.valid),
    hijacked_utility_pass_rate: safeRate(counts.hijacked_and_utility_passed, counts.hijacked_valid),
    resisted_utility_pass_rate: safeRate(counts.resisted_and_utility_passed, counts.resisted_valid),
  };
  const trajectory = {
    duration_seconds_mean: mean(records.map((record) => record.duration_seconds)),
    duration_seconds_p50: percentile(records.map((record) => record.duration_seconds), 0.5),
    duration_seconds_p95: percentile(records.map((record) => record.duration_seconds), 0.95),
    message_count_mean: mean(records.map((record) => record.message_count)),
    message_count_p50: percentile(records.map((record) => record.message_count), 0.5),
    message_count_p95: percentile(records.map((record) => record.message_count), 0.95),
    tool_call_count_mean: mean(records.map((record) => record.tool_call_count)),
    tool_call_count_p50: percentile(records.map((record) => record.tool_call_count), 0.5),
    tool_call_count_p95: percentile(records.map((record) => record.tool_call_count), 0.95),
    tool_error_count_total: records.reduce((sum, record) => sum + record.tool_error_count, 0),
    injection_site_count_mean: mean(records.map((record) => record.injection_site_count)),
    raw_trace_bytes_mean: mean(records.map((record) => record.raw_trace_bytes)),
    raw_trace_bytes_total: records.reduce((sum, record) => sum + record.raw_trace_bytes, 0),
  };
  return { counts, rates, trajectory };
}

function groupBy(records, keyFunction) {
  const grouped = new Map();
  for (const record of records) {
    const key = keyFunction(record);
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(record);
  }
  return grouped;
}

async function loadPairManifest(experimentRoot) {
  const manifestRoot = path.join(experimentRoot, "manifests");
  if (!(await exists(manifestRoot))) return null;
  const candidates = (await readdir(manifestRoot, { withFileTypes: true }))
    .filter((entry) => entry.isFile() && entry.name.endsWith(".json"))
    .map((entry) => path.join(manifestRoot, entry.name))
    .sort();
  if (candidates.length === 0) return null;
  if (candidates.length > 1) {
    throw new Error(`Multiple pair manifests found; specify a result root with one manifest: ${candidates.join(", ")}`);
  }
  const manifestPath = candidates[0];
  const manifest = readJson(await readFile(manifestPath, "utf8"), manifestPath);
  if (manifest.schema_version !== "sd_guard.task_injection_pair_manifest.v1") {
    throw new Error(`Unsupported pair manifest schema in ${manifestPath}: ${manifest.schema_version}`);
  }
  return { path: manifestPath, value: manifest };
}

function buildCoverage(records, attacks, suites, pairManifest) {
  const usersBySuite = new Map();
  const injectionsBySuite = new Map();
  for (const suite of suites) {
    usersBySuite.set(suite, new Set());
    injectionsBySuite.set(suite, new Set());
  }
  for (const record of records) {
    if (!usersBySuite.has(record.suite_name)) usersBySuite.set(record.suite_name, new Set());
    if (!injectionsBySuite.has(record.suite_name)) injectionsBySuite.set(record.suite_name, new Set());
    if (record.user_task_id) usersBySuite.get(record.suite_name).add(record.user_task_id);
    if (record.injection_task_id) injectionsBySuite.get(record.suite_name).add(record.injection_task_id);
  }

  const observed = new Map();
  for (const record of records) {
    const groupKey = `${record.attack_type}\u0000${record.suite_name}`;
    if (!observed.has(groupKey)) observed.set(groupKey, new Set());
    observed.get(groupKey).add(`${record.user_task_id}\u0000${record.injection_task_id}`);
  }

  const rows = [];
  let expectedTotal = 0;
  for (const attack of attacks) {
    for (const suite of suites) {
      const users = usersBySuite.get(suite) ?? new Set();
      const injections = injectionsBySuite.get(suite) ?? new Set();
      const manifestSuite = pairManifest?.value?.suites?.[suite] ?? null;
      const expected = manifestSuite
        ? Array.isArray(manifestSuite.sampled_pairs)
          ? manifestSuite.sampled_pairs.length
          : 0
        : users.size * injections.size;
      const observedCount = observed.get(`${attack}\u0000${suite}`)?.size ?? 0;
      expectedTotal += expected;
      rows.push({
        attack_type: attack,
        suite_name: suite,
        user_tasks: manifestSuite?.pool_user_tasks ?? users.size,
        injection_tasks: manifestSuite?.pool_injection_tasks ?? injections.size,
        expected_combinations: expected,
        observed_combinations: observedCount,
        missing_combinations: Math.max(0, expected - observedCount),
        coverage_rate: safeRate(observedCount, expected),
      });
    }
  }
  return {
    mode: pairManifest ? "sample_manifest" : "full_cross_product",
    manifest_path: pairManifest?.path ?? null,
    expected_total: expectedTotal,
    observed_total: records.length,
    complete: rows.every((row) => row.missing_combinations === 0),
    rows,
  };
}

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  const text = typeof value === "boolean" ? (value ? "true" : "false") : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function toCsv(rows, columns) {
  const lines = [columns.join(",")];
  for (const row of rows) {
    lines.push(columns.map((column) => csvEscape(row[column])).join(","));
  }
  return `\uFEFF${lines.join("\r\n")}\r\n`;
}

function flattenGroupRow(level, attack, suite, summary) {
  return {
    group_level: level,
    attack_type: attack ?? "",
    suite_name: suite ?? "",
    ...summary.counts,
    ...summary.rates,
    ...summary.trajectory,
  };
}

function pct(value) {
  return value === null || value === undefined ? "—" : `${(value * 100).toFixed(2)}%`;
}

function decimal(value, digits = 2) {
  return value === null || value === undefined ? "—" : Number(value).toFixed(digits);
}

function markdownMetricsRow(label, summary) {
  return `| ${label} | ${summary.counts.total} | ${summary.counts.valid} | ${pct(summary.rates.valid_rate)} | ${summary.counts.payload_present_valid} | ${summary.counts.hijacked_valid} | ${pct(summary.rates.ASR)} | ${summary.counts.utility_passed_valid} | ${pct(summary.rates.utility_pass_rate_among_valid)} |`;
}

function buildMarkdown({
  analysisStatus,
  generatedAt,
  tracesRoot,
  integrity,
  coverage,
  overall,
  attacks,
  suites,
  byAttack,
  bySuite,
  byAttackSuite,
  invalidReasons,
  provenance,
}) {
  const lines = [];
  lines.push("# AgentDojo Task-Injection Trace Statistics", "");
  lines.push(`- Generated: \`${generatedAt}\``);
  lines.push(`- Trace root: \`${tracesRoot}\``);
  lines.push(`- Parsed traces: **${integrity.parsed_samples}**`);
  lines.push(`- Unparsed trace directories: **${integrity.unparsed_trace_directories}**`);
  lines.push(`- Integrity errors: **${integrity.errors.length}**`);
  lines.push(`- Hashes recomputed: **${integrity.hashes_checked}**`);
  lines.push(`- Coverage complete: **${coverage.complete ? "yes" : "no"}**`, "");
  if (analysisStatus !== "complete") {
    lines.push(
      "> [!WARNING]",
      "> This is a partial diagnostic summary. Do not use its rates as final experimental results until all integrity errors are resolved.",
      "",
    );
  }

  lines.push("## Overall", "");
  lines.push("| Group | Total | Valid | Valid rate | Payload present | Hijacked | ASR | Utility passed | Utility pass rate |");
  lines.push("|---|---:|---:|---:|---:|---:|---:|---:|---:|");
  lines.push(markdownMetricsRow("All traces", overall), "");

  lines.push("## By attack", "");
  lines.push("| Attack | Total | Valid | Valid rate | Payload present | Hijacked | ASR | Utility passed | Utility pass rate |");
  lines.push("|---|---:|---:|---:|---:|---:|---:|---:|---:|");
  for (const attack of attacks) lines.push(markdownMetricsRow(`\`${attack}\``, byAttack[attack]));
  lines.push("");

  lines.push("## By suite", "");
  lines.push("| Suite | Total | Valid | Valid rate | Payload present | Hijacked | ASR | Utility passed | Utility pass rate |");
  lines.push("|---|---:|---:|---:|---:|---:|---:|---:|---:|");
  for (const suite of suites) lines.push(markdownMetricsRow(`\`${suite}\``, bySuite[suite]));
  lines.push("");

  lines.push("## Attack × suite", "");
  lines.push("| Attack / suite | Total | Valid | Valid rate | Payload present | Hijacked | ASR | Utility passed | Utility pass rate |");
  lines.push("|---|---:|---:|---:|---:|---:|---:|---:|---:|");
  for (const attack of attacks) {
    for (const suite of suites) {
      const key = `${attack}/${suite}`;
      if (byAttackSuite[key]) lines.push(markdownMetricsRow(`\`${key}\``, byAttackSuite[key]));
    }
  }
  lines.push("");

  lines.push("## Joint outcomes", "");
  lines.push("| Outcome among valid traces | Count | Share of valid traces |");
  lines.push("|---|---:|---:|");
  const jointRows = [
    ["Hijacked + utility passed", overall.counts.hijacked_and_utility_passed],
    ["Hijacked + utility failed", overall.counts.hijacked_and_utility_failed],
    ["Resisted + utility passed", overall.counts.resisted_and_utility_passed],
    ["Resisted + utility failed", overall.counts.resisted_and_utility_failed],
  ];
  for (const [label, count] of jointRows) {
    lines.push(`| ${label} | ${count} | ${pct(safeRate(count, overall.counts.valid))} |`);
  }
  lines.push("");

  lines.push("## Trajectory characteristics", "");
  lines.push("| Metric | Mean | P50 | P95 |");
  lines.push("|---|---:|---:|---:|");
  lines.push(`| Duration (seconds) | ${decimal(overall.trajectory.duration_seconds_mean)} | ${decimal(overall.trajectory.duration_seconds_p50)} | ${decimal(overall.trajectory.duration_seconds_p95)} |`);
  lines.push(`| Messages | ${decimal(overall.trajectory.message_count_mean)} | ${decimal(overall.trajectory.message_count_p50)} | ${decimal(overall.trajectory.message_count_p95)} |`);
  lines.push(`| Tool calls | ${decimal(overall.trajectory.tool_call_count_mean)} | ${decimal(overall.trajectory.tool_call_count_p50)} | ${decimal(overall.trajectory.tool_call_count_p95)} |`);
  lines.push("");

  lines.push("## Coverage", "");
  lines.push(`Coverage mode: \`${coverage.mode}\``, "");
  lines.push("| Attack | Suite | User tasks | Injection tasks | Expected | Observed | Missing | Coverage |");
  lines.push("|---|---|---:|---:|---:|---:|---:|---:|");
  for (const row of coverage.rows) {
    lines.push(`| \`${row.attack_type}\` | \`${row.suite_name}\` | ${row.user_tasks} | ${row.injection_tasks} | ${row.expected_combinations} | ${row.observed_combinations} | ${row.missing_combinations} | ${pct(row.coverage_rate)} |`);
  }
  lines.push("");

  lines.push("## Invalid reasons", "");
  if (Object.keys(invalidReasons).length === 0) {
    lines.push("No invalid traces.", "");
  } else {
    lines.push("| Reason | Count |", "|---|---:|");
    for (const [reason, count] of Object.entries(invalidReasons)) lines.push(`| ${reason} | ${count} |`);
    lines.push("");
  }

  lines.push("## Provenance", "");
  for (const [field, counts] of Object.entries(provenance)) {
    lines.push(`- **${field}:** ${Object.entries(counts).map(([value, count]) => `\`${value}\` (${count})`).join(", ")}`);
  }
  lines.push("");

  lines.push("## Definitions", "");
  lines.push("- **Valid:** `ground_truth.valid=true`.");
  lines.push("- **Hijacked:** `ground_truth.hijacked=true`; AgentDojo reports that the injection goal executed.");
  lines.push("- **ASR:** hijacked, payload-present, valid traces divided by all payload-present, valid traces.");
  lines.push("- **Utility pass rate:** valid traces with `utility_passed=true` divided by all valid traces.");
  lines.push("- Percentages in grouped tables use the same denominators within that group.", "");
  return `${lines.join("\n")}\n`;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const { experimentRoot, tracesRoot, outputRoot } = await resolveRoots(args.input, args.output);
  const entries = (await readdir(tracesRoot, { withFileTypes: true }))
    .filter((entry) => entry.isDirectory())
    .map((entry) => ({ ...entry, parentPath: tracesRoot }))
    .sort((a, b) => a.name.localeCompare(b.name));
  if (entries.length === 0) throw new Error(`No trace directories found under ${tracesRoot}`);

  const integrity = {
    trace_directories: entries.length,
    parsed_samples: 0,
    unparsed_trace_directories: 0,
    hashes_checked: 0,
    duplicate_sample_ids: 0,
    duplicate_task_combinations: 0,
    zero_byte_files: {
      "ground_truth.json": 0,
      "raw_trace.json": 0,
      "raw_trace.sha256": 0,
    },
    errors: [],
  };
  const records = [];
  for (let index = 0; index < entries.length; index += 1) {
    const record = await loadTraceDirectory(entries[index], args.verifyHashes, integrity);
    if (record) records.push(record);
    if ((index + 1) % 500 === 0 || index + 1 === entries.length) {
      console.log(`Loaded ${index + 1}/${entries.length} trace directories`);
    }
  }
  integrity.parsed_samples = records.length;
  integrity.unparsed_trace_directories = entries.length - records.length;

  const sampleIds = new Set();
  const taskCombinations = new Set();
  for (const record of records) {
    if (sampleIds.has(record.sample_id)) integrity.duplicate_sample_ids += 1;
    sampleIds.add(record.sample_id);
    const combination = [
      record.attack_type,
      record.suite_name,
      record.user_task_id,
      record.injection_task_id,
    ].join("\u0000");
    if (taskCombinations.has(combination)) integrity.duplicate_task_combinations += 1;
    taskCombinations.add(combination);
  }

  const attacks = preferredSort(new Set(records.map((record) => record.attack_type)), ATTACK_ORDER);
  const suites = preferredSort(new Set(records.map((record) => record.suite_name)), SUITE_ORDER);
  const overall = aggregate(records);
  const byAttack = {};
  for (const [attack, group] of groupBy(records, (record) => record.attack_type)) {
    byAttack[attack] = aggregate(group);
  }
  const bySuite = {};
  for (const [suite, group] of groupBy(records, (record) => record.suite_name)) {
    bySuite[suite] = aggregate(group);
  }
  const byAttackSuite = {};
  for (const [key, group] of groupBy(records, (record) => `${record.attack_type}/${record.suite_name}`)) {
    byAttackSuite[key] = aggregate(group);
  }

  const invalidReasonMap = new Map();
  for (const record of records.filter((item) => item.valid === false)) {
    addCount(invalidReasonMap, record.invalid_reason ?? "<unspecified>");
  }
  const invalidReasons = mapToSortedObject(invalidReasonMap);

  const provenanceMaps = {
    benchmark_versions: new Map(),
    run_ids: new Map(),
    seeds: new Map(),
    model_ids: new Map(),
    pipeline_names: new Map(),
  };
  for (const record of records) {
    addCount(provenanceMaps.benchmark_versions, record.benchmark_version);
    addCount(provenanceMaps.run_ids, record.run_id);
    addCount(provenanceMaps.seeds, record.seed);
    addCount(provenanceMaps.model_ids, record.model_id);
    addCount(provenanceMaps.pipeline_names, record.pipeline_name);
  }
  const provenance = Object.fromEntries(
    Object.entries(provenanceMaps).map(([key, value]) => [key, mapToSortedObject(value)]),
  );
  const pairManifest = await loadPairManifest(experimentRoot);
  const coverage = buildCoverage(records, attacks, suites, pairManifest);
  const generatedAt = new Date().toISOString();
  const analysisStatus =
    integrity.errors.length === 0 &&
    integrity.duplicate_sample_ids === 0 &&
    integrity.duplicate_task_combinations === 0 &&
    coverage.complete
      ? "complete"
      : "incomplete";

  const summary = {
    schema_version: "sd_guard.agentdojo_trace_statistics.v1",
    analysis_status: analysisStatus,
    generated_at: generatedAt,
    experiment_root: experimentRoot,
    traces_root: tracesRoot,
    integrity,
    coverage,
    provenance,
    overall,
    attacks,
    suites,
    by_attack: byAttack,
    by_suite: bySuite,
    by_attack_suite: byAttackSuite,
    invalid_reasons: invalidReasons,
    definitions: {
      valid: "ground_truth.valid=true",
      hijacked: "ground_truth.hijacked=true",
      ASR: "hijacked payload-present valid traces / payload-present valid traces",
      utility_pass_rate: "utility-passed valid traces / valid traces",
    },
  };

  const groupRows = [flattenGroupRow("overall", null, null, overall)];
  for (const attack of attacks) groupRows.push(flattenGroupRow("attack", attack, null, byAttack[attack]));
  for (const suite of suites) groupRows.push(flattenGroupRow("suite", null, suite, bySuite[suite]));
  for (const attack of attacks) {
    for (const suite of suites) {
      const key = `${attack}/${suite}`;
      if (byAttackSuite[key]) {
        groupRows.push(flattenGroupRow("attack_suite", attack, suite, byAttackSuite[key]));
      }
    }
  }

  const recordColumns = [
    "sample_id",
    "suite_name",
    "attack_type",
    "user_task_id",
    "injection_task_id",
    "valid",
    "invalid_reason",
    "payload_present",
    "hijacked",
    "utility_passed",
    "raw_agentdojo_security",
    "duration_seconds",
    "message_count",
    "assistant_message_count",
    "tool_message_count",
    "tool_call_count",
    "tool_error_count",
    "injection_site_count",
    "raw_trace_bytes",
    "evaluation_timestamp",
    "run_error",
    "run_id",
    "seed",
    "benchmark_version",
    "model_id",
    "pipeline_name",
    "trace_sha256",
  ];
  const groupColumns = Object.keys(groupRows[0]);

  await mkdir(outputRoot, { recursive: true });
  await Promise.all([
    writeFile(path.join(outputRoot, "trace_statistics.json"), `${JSON.stringify(summary, null, 2)}\n`, "utf8"),
    writeFile(
      path.join(outputRoot, "trace_statistics.md"),
      buildMarkdown({
        analysisStatus,
        generatedAt,
        tracesRoot,
        integrity,
        coverage,
        overall,
        attacks,
        suites,
        byAttack,
        bySuite,
        byAttackSuite,
        invalidReasons,
        provenance,
      }),
      "utf8",
    ),
    writeFile(path.join(outputRoot, "group_statistics.csv"), toCsv(groupRows, groupColumns), "utf8"),
    writeFile(path.join(outputRoot, "trace_records.csv"), toCsv(records, recordColumns), "utf8"),
  ]);

  console.log("");
  console.log(`Analysis status:  ${analysisStatus}`);
  console.log(`Parsed traces:    ${records.length}`);
  console.log(`Valid traces:     ${overall.counts.valid}`);
  console.log(`Hijacked traces:  ${overall.counts.hijacked_valid}`);
  console.log(`ASR:              ${pct(overall.rates.ASR)}`);
  console.log(`Utility pass rate:${overall.rates.utility_pass_rate_among_valid === null ? " —" : ` ${pct(overall.rates.utility_pass_rate_among_valid)}`}`);
  console.log(`Integrity errors: ${integrity.errors.length}`);
  console.log(`Output:           ${outputRoot}`);
  if (analysisStatus !== "complete") {
    process.exitCode = 2;
  }
}

main().catch((error) => {
  console.error(`ERROR: ${error.message}`);
  process.exitCode = 1;
});
