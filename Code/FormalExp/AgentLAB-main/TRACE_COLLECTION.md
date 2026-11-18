# AgentLAB raw trace collection

The four non-AgentDojo runners now write the same immutable artifact envelope
used by the Task Injection export:

```text
<trace-dir>/
  agentlab-<stable-id>/
    raw_trace.json
    raw_trace.sha256
```

The schema is `sd_guard.raw_trace.v1`. These files intentionally contain raw
runtime facts and `label_status: pending`; episode and event labels can be
derived later without rerunning the victim model.

The commands below assume an OpenAI-compatible vLLM server for
`Qwen/Qwen3-8B` at `http://localhost:8001/v1`. Adjust the model ID and URL to
match the server used for Task Injection.

## Intent Hijacking

```powershell
python .\Intent-Hijacking.py `
  --victim Qwen/Qwen3-8B `
  --victim_url http://localhost:8001/v1 `
  --dataset .\data\filtered_top_200_attacks.json `
  --num_samples 10 `
  --sequential `
  --output_dir .\runs\intent-smoke `
  --trace_dir .\runs\intent-smoke\traces
```

Saved victim data includes the system prompt, every attacker/victim turn,
unmodified local-model content, and `reasoning_content` when vLLM exposes it.

## Tool Chaining

```powershell
python .\Tool-chaining.py `
  --victim_model Qwen/Qwen3-8B `
  --victim_base_url http://localhost:8001/v1 `
  --dataset .\data\filtered_top_200_attacks.json `
  --num_samples 10 `
  --output_dir .\runs\tool-smoke `
  --trace_dir .\runs\tool-smoke\traces
```

Saved victim data includes the complete conversation, tool schemas, proposed
tool calls, tool results, executed tools, and planner/judge records.

## Objective Drifting

Use the v12 script directly so the 8B model ID is not replaced by a top-level
victim preset:

```powershell
python .\attacks\goal_drift\goal_drift_v12.py `
  --victim_model Qwen/Qwen3-8B `
  --victim_base_url http://localhost:8001/v1 `
  --victim_api_key EMPTY `
  --num_tasks 10 `
  --num_seeds 1 `
  --output .\runs\objective-smoke\results.json `
  --trace_dir .\runs\objective-smoke\traces `
  --verbose
```

Saved victim data includes each exact request window and raw response, original
and perturbed WebShop observations, injections, parsed actions, next
observations, prices, and termination state. The trace records whether the
real WebShop bridge or the fallback simulator was used.

## Memory Poisoning

```powershell
python .\Memory-Poisoning.py `
  --targets local `
  --local_model Qwen/Qwen3-8B `
  --local_url http://localhost:8001/v1 `
  --data_path .\data\all_refused_combined_200.json `
  --target_samples 10 `
  --sequential `
  --trace_dir .\runs\memory-smoke\traces
```

Both baseline and every poisoning attempt are saved. Each attack trace contains
the written poison memories, actually retrieved memories, final system prompt,
tool schemas, complete message/tool trajectory, and final response.

## Recommended smoke test

Start with one or two samples in every track. For each trace, verify that:

1. `target.model_id` is the intended Qwen3-8B endpoint;
2. `trajectory.messages` contains the complete victim interaction;
3. track-specific observations, tool results, or retrieved memories are present;
4. `raw_trace.sha256` matches the JSON file;
5. Objective Drifting reports the intended WebShop backend.

