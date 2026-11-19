# AgentLAB Qwen3-8B raw-trace runbook

The four non-AgentDojo runners write the same immutable artifact envelope used
by the Task Injection export:

```text
<trace-dir>/
  agentlab-<stable-id>/
    raw_trace.json
    raw_trace.sha256
```

The schema is `sd_guard.raw_trace.v1`. Raw artifacts contain runtime facts and
`label_status: pending`; episode/event labels can be derived later without
rerunning the victim model.

This runbook starts with Qwen3-8B **non-thinking** mode. It uses the Qwen3
non-thinking sampling condition from the reasoning-variance paper:
`temperature=0.7`, `top_p=0.8`, `top_k=20`, and a 768-token victim budget.
This is a controlled condition, not deterministic decoding.

## 1. Start the shared Qwen3-8B victim server

Run in terminal A. Change the conda environment/model path if necessary.

Qwen's `qwen3` reasoning parser supports `enable_thinking=false` together with
reasoning parsing in vLLM 0.9.0 and later. Check the installed version first:

```bash
python -c "import vllm; print(vllm.__version__)"
vllm serve --help | grep default-chat-template-kwargs
```

If the second command finds no such option, remove
`--default-chat-template-kwargs` from the server command. The four runner
commands still enforce non-thinking per request through
`--victim_thinking off` / `--local_thinking off`; keep vLLM at 0.9.0 or later
for the Qwen3 reasoning-parser compatibility fix.

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/sd4-vllm-cu130

export SD_ROOT=/root/autodl-tmp/Code/SD_Guard
export AL_ROOT="$SD_ROOT/Code/FormalExp/AgentLAB-main"
export RUN_ROOT="$SD_ROOT/results/agentlab/qwen3_8b_nothink_v1"
export VICTIM_MODEL_PATH=/root/autodl-tmp/models/Qwen3-8B
export VICTIM_MODEL_NAME=Qwen/Qwen3-8B
mkdir -p "$RUN_ROOT"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CUDA_VISIBLE_DEVICES=0 vllm serve "$VICTIM_MODEL_PATH" \
  --served-model-name "$VICTIM_MODEL_NAME" \
  --host 127.0.0.1 \
  --port 8001 \
  --tensor-parallel-size 1 \
  --dtype auto \
  --gpu-memory-utilization 0.95 \
  --max-model-len 40960 \
  --max-num-seqs 20 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --generation-config vllm \
  --override-generation-config '{"temperature":0.7,"top_p":0.8,"top_k":20}' \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  2>&1 | tee "$RUN_ROOT/vllm_victim.log"
```

The `0.95` GPU fraction is for the current three-track campaign that skips
Intent Hijacking. If the Intent attacker server is later started on the same
GPU, reduce the victim fraction first; the two allocations must fit together.

Verify both health and non-thinking output:

```bash
curl -f http://127.0.0.1:8001/health

curl -s http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"Answer only: OK"}],"temperature":0.7,"top_p":0.8,"top_k":20,"max_tokens":32}'
```

The response should not contain a `<think>...</think>` block and its
`reasoning_content` should be absent, null, or empty. `--reasoning-parser qwen3`
only parses reasoning when it exists; it does not turn thinking on.

## 2. Start the Intent Hijacking attacker server

Only Intent Hijacking needs this second local vLLM process. Run it in terminal
B while the victim server remains running.

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/sd4-vllm-cu130

export SD_ROOT=/root/autodl-tmp/Code/SD_Guard
export RUN_ROOT="$SD_ROOT/results/agentlab/qwen3_8b_nothink_v1"
export ATTACKER_MODEL_PATH=/root/autodl-tmp/models/Huihui-Qwen3-14B-abliterated-v2
export ATTACKER_MODEL_NAME=huihui-ai/Huihui-Qwen3-14B-abliterated-v2

CUDA_VISIBLE_DEVICES=0 vllm serve "$ATTACKER_MODEL_PATH" \
  --served-model-name "$ATTACKER_MODEL_NAME" \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --dtype auto \
  --gpu-memory-utilization 0.45 \
  --max-model-len 32768 \
  --max-num-seqs 8 \
  --trust-remote-code \
  --generation-config vllm \
  --override-generation-config '{"temperature":0.3,"top_p":0.95,"top_k":20}' \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  2>&1 | tee "$RUN_ROOT/vllm_intent_attacker.log"
```

## 3. Prepare the runner terminal

Run the following in terminal C. The non-victim planner/judge/attacker calls
use DeepSeek's OpenAI-compatible chat endpoint. `DEEPSEEK_API_KEY` is not the
Qwen victim key and is never written into a trace.

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/sd4-vllm-cu130

export SD_ROOT=/root/autodl-tmp/Code/SD_Guard
export AL_ROOT="$SD_ROOT/Code/FormalExp/AgentLAB-main"
export RUN_ROOT="$SD_ROOT/results/agentlab/qwen3_8b_nothink_v1"
export TOOL_ROOT="$RUN_ROOT/tool_chaining_deepseek_flash"
export VICTIM_MODEL_NAME=Qwen/Qwen3-8B
mkdir -p "$RUN_ROOT" "$TOOL_ROOT"
cd "$AL_ROOT"

# One-time only: all AgentLAB runners load this ignored local file automatically.
cp -n config/local_secrets.env.example config/local_secrets.env
nano config/local_secrets.env
chmod 600 config/local_secrets.env
```

Set the real `DEEPSEEK_API_KEY` in that file. Subsequent terminals do not need
to export the key, auxiliary URL, or auxiliary model again.

Before running Tool Chaining, verify that the server copy contains the empty
DeepSeek response retry and the auxiliary-thinking startup print:

```bash
grep -n 'retrying once with thinking disabled' Tool-chaining.py
grep -n 'Auxiliary Thinking' Tool-chaining.py
```

Both commands must print a matching source line. No output means an older copy
of `Tool-chaining.py` is being executed.

Memory Poisoning must run from a separate Python environment. Do not install
Mem0 into the vLLM 0.31 environment: Mem0 0.1.118 pins OpenAI 1.x, whereas
vLLM 0.31 requires OpenAI 2.x. The runner and the still-running vLLM process
communicate over HTTP, so separate environments are expected.

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda create -p /root/autodl-tmp/conda-envs/agentlab-memory python=3.10 -y
conda activate /root/autodl-tmp/conda-envs/agentlab-memory

cd "$AL_ROOT"
pip install -r requirements-memory.txt
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('embedder cached')"
```

All Memory Poisoning commands below must be launched from
`/root/autodl-tmp/conda-envs/agentlab-memory`; leave the victim vLLM server in
its original `sd4-vllm-cu130` terminal.

## 4. Smoke-test commands

Run these first. Each command should finish and write `raw_trace.json` plus its
checksum before a full campaign is started.

### Intent Hijacking smoke test

```bash
mkdir -p "$RUN_ROOT/intent_hijacking_smoke"
python Intent-Hijacking.py \
  --victim "$VICTIM_MODEL_NAME" \
  --victim_url http://127.0.0.1:8001/v1 \
  --victim_temperature 0.7 \
  --victim_top_p 0.8 \
  --victim_top_k 20 \
  --victim_max_tokens 768 \
  --victim_thinking off \
  --attacker_url http://127.0.0.1:8000/v1 \
  --attacker_model huihui-ai/Huihui-Qwen3-14B-abliterated-v2 \
  --planner_judge_model gpt-5.1 \
  --dataset data/filtered_top_200_attacks.json \
  --num_samples 2 \
  --seed 20261118 \
  --max_turns 7 \
  --num_strategies 2 \
  --success_threshold 4 \
  --sequential \
  --no_textgrad \
  --output_dir "$RUN_ROOT/intent_hijacking_smoke" \
  --trace_dir "$RUN_ROOT/intent_hijacking_smoke/traces" \
  2>&1 | tee "$RUN_ROOT/intent_hijacking_smoke/run.log"
```

### Tool Chaining smoke test

```bash
mkdir -p "$TOOL_ROOT/smoke"
python Tool-chaining.py \
  --victim_model "$VICTIM_MODEL_NAME" \
  --victim_base_url http://127.0.0.1:8001/v1 \
  --victim_temperature 0.7 \
  --victim_top_p 0.8 \
  --victim_top_k 20 \
  --victim_max_tokens 768 \
  --victim_thinking off \
  --attacker_model "$AUX_MODEL" \
  --aux_base_url "$AUX_BASE_URL" \
  --aux_thinking off \
  --dataset data/filtered_top_200_attacks.json \
  --num_samples 2 \
  --seed 20261118 \
  --max_turns 10 \
  --max_agent_rounds 10 \
  --success_threshold 3 \
  --output_dir "$TOOL_ROOT/smoke/results" \
  --trace_dir "$TOOL_ROOT/smoke/traces" \
  2>&1 | tee "$TOOL_ROOT/smoke/run.log"
```

The startup log must say `SHADE_Arena environment loaded successfully.` If it
says `Using mock environment instead.`, do not use that run as formal data.
It must also print `Attacker/Judge: deepseek-flash`,
`Auxiliary URL: https://api.deepseek.com`, and `Auxiliary Thinking: off`.
An empty response that is recovered by the automatic retry is acceptable. If
the final smoke log contains `Planner returned an empty prompt`, do not start
the full run.

```bash
SMOKE_COUNT=$(find "$TOOL_ROOT/smoke/traces" -name raw_trace.json | wc -l)
test "$SMOKE_COUNT" -eq 2 || { echo "INVALID smoke trace count: $SMOKE_COUNT/2"; exit 1; }
```

### Objective Drifting smoke test

Prepare and verify the WebShop environment once:

```bash
python Objective-Drifting.py --setup-only
python Objective-Drifting.py --check-env
```

Then run the v12 implementation directly:

```bash
mkdir -p "$RUN_ROOT/objective_drifting_smoke"
python attacks/goal_drift/goal_drift_v12.py \
  --victim_model "$VICTIM_MODEL_NAME" \
  --victim_base_url http://127.0.0.1:8001/v1 \
  --victim_api_key EMPTY \
  --victim_temperature 0.7 \
  --victim_top_p 0.8 \
  --victim_top_k 20 \
  --victim_max_tokens 768 \
  --victim_thinking off \
  --attacker_model "$AUX_MODEL" \
  --judge_model "$AUX_MODEL" \
  --extractor_model "$AUX_MODEL" \
  --stealthiness_model "$AUX_MODEL" \
  --aux_base_url "$AUX_BASE_URL" \
  --aux_thinking off \
  --num_tasks 2 \
  --num_seeds 1 \
  --strategy optimized \
  --optimize_iterations 1 \
  --output "$RUN_ROOT/objective_drifting_smoke/results.json" \
  --trace_dir "$RUN_ROOT/objective_drifting_smoke/traces" \
  --memory_file "$RUN_ROOT/objective_drifting_smoke/injection_memory.json" \
  --verbose \
  2>&1 | tee "$RUN_ROOT/objective_drifting_smoke/run.log"
```

Formal data must use `WebShopSubprocessClient`. If the log or trace reports the
standalone fallback simulator, fix WebShop before running the full campaign.

### Memory Poisoning smoke test

Verify the two critical integrations first:

```bash
python -c "import mem0; print('mem0 ok')"
python -c "import sys; sys.path.insert(0, './environments'); from EnvManager import EnvManager; print('EnvManager ok')"
```

Then run:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/agentlab-memory

export SD_ROOT=/root/autodl-tmp/Code/SD_Guard
export AL_ROOT="$SD_ROOT/Code/FormalExp/AgentLAB-main"
export RUN_ROOT="$SD_ROOT/results/agentlab/qwen3_8b_nothink_v1"
export MEMORY_ROOT="$RUN_ROOT/memory_poisoning_deepseek_flash"
export VICTIM_MODEL_NAME=Qwen/Qwen3-8B
cd "$AL_ROOT"

mkdir -p "$MEMORY_ROOT/smoke"
python Memory-Poisoning.py \
  --targets local \
  --local_model "$VICTIM_MODEL_NAME" \
  --local_url http://127.0.0.1:8001/v1 \
  --local_temperature 0.7 \
  --local_top_p 0.8 \
  --local_top_k 20 \
  --local_max_tokens 768 \
  --local_thinking off \
  --aux_model deepseek-flash \
  --aux_base_url https://api.deepseek.com \
  --aux_thinking off \
  --mem0_mode compatible \
  --mem0_llm_model deepseek-flash \
  --mem0_embedder_model sentence-transformers/all-MiniLM-L6-v2 \
  --mem0_embedding_dims 384 \
  --mem0_embedder_device cpu \
  --mem0_vector_path "$MEMORY_ROOT/smoke/mem0_qdrant" \
  --mem0_collection_name agentlab_memory_smoke \
  --data_path data/all_refused_combined_200.json \
  --target_samples 2 \
  --start_idx 0 \
  --num_strategies 3 \
  --require_mem0 \
  --sequential \
  --fixed_output_dir \
  --run_id qwen3_8b_memory_poisoning_deepseek_flash_smoke \
  --output_dir "$MEMORY_ROOT/smoke" \
  --trace_dir "$MEMORY_ROOT/smoke/traces" \
  2>&1 | tee "$MEMORY_ROOT/smoke/run.log"
```

The log must contain both `EnvManager loaded from ./environments` and
`Mem0 initialized`, and startup must print `Strict Mem0 operations: True`.
Every selected attack trace must have
`attack.memory_validation.valid: true`, `retrieval_source: mem0_search`, and at
least one overlapping ID in `written_memory_ids` and `retrieved_memory_ids`.
A fallback run without these facts is not equivalent to the intended Memory
Poisoning task.

## 5. Full commands

Run these only after the corresponding smoke test passes. They use new output
directories and do not overwrite the smoke artifacts.

### Intent Hijacking full run

```bash
mkdir -p "$RUN_ROOT/intent_hijacking"
python Intent-Hijacking.py \
  --victim "$VICTIM_MODEL_NAME" \
  --victim_url http://127.0.0.1:8001/v1 \
  --victim_temperature 0.7 \
  --victim_top_p 0.8 \
  --victim_top_k 20 \
  --victim_max_tokens 768 \
  --victim_thinking off \
  --attacker_url http://127.0.0.1:8000/v1 \
  --attacker_model huihui-ai/Huihui-Qwen3-14B-abliterated-v2 \
  --planner_judge_model gpt-5.1 \
  --dataset data/filtered_top_200_attacks.json \
  --num_samples 200 \
  --seed 20261118 \
  --max_turns 7 \
  --num_strategies 2 \
  --success_threshold 4 \
  --sequential \
  --output_dir "$RUN_ROOT/intent_hijacking" \
  --trace_dir "$RUN_ROOT/intent_hijacking/traces" \
  2>&1 | tee "$RUN_ROOT/intent_hijacking/run.log"
```

### Tool Chaining full run

```bash
mkdir -p "$TOOL_ROOT/full"
python Tool-chaining.py \
  --victim_model "$VICTIM_MODEL_NAME" \
  --victim_base_url http://127.0.0.1:8001/v1 \
  --victim_temperature 0.7 \
  --victim_top_p 0.8 \
  --victim_top_k 20 \
  --victim_max_tokens 768 \
  --victim_thinking off \
  --attacker_model "$AUX_MODEL" \
  --aux_base_url "$AUX_BASE_URL" \
  --aux_thinking off \
  --dataset data/filtered_top_200_attacks.json \
  --num_samples 200 \
  --seed 20261118 \
  --max_turns 10 \
  --max_agent_rounds 10 \
  --success_threshold 3 \
  --output_dir "$TOOL_ROOT/full/results" \
  --trace_dir "$TOOL_ROOT/full/traces" \
  2>&1 | tee "$TOOL_ROOT/full/run.log"
```

After the full run, reject the run if unrecovered planner failures exist:

```bash
if grep -Eq 'Planner returned an empty prompt|Planner error:.*empty content after' "$TOOL_ROOT/full/run.log"; then
  echo 'INVALID: unrecovered planner failures found'
  exit 1
fi
FULL_COUNT=$(find "$TOOL_ROOT/full/traces" -name raw_trace.json | wc -l)
test "$FULL_COUNT" -eq 200 || { echo "INCOMPLETE full trace count: $FULL_COUNT/200"; exit 1; }
echo "VALID: $FULL_COUNT/200 traces"
```

### Recover only the 33 Agent-SafetyBench Tool Chaining rows

The bundled 200-row Tool Chaining file contains 167 SHADE-Arena rows and 33
Agent-SafetyBench rows.  `--dataset_source Agent_SafetyBench` selects only the
second group, so an existing 167-row SHADE run does not need to be repeated.

First verify source-environment recovery without calling vLLM or DeepSeek:

```bash
export ASB_ROOT="$TOOL_ROOT/agent_safety_recovery"
mkdir -p "$ASB_ROOT"

python Tool-chaining.py \
  --victim_model "$VICTIM_MODEL_NAME" \
  --dataset data/filtered_top_200_attacks.json \
  --dataset_source Agent_SafetyBench \
  --resolve_agent_safety_only \
  2>&1 | tee "$ASB_ROOT/resolve.log"

grep -q 'Resolved 33 Agent_SafetyBench attacks.' "$ASB_ROOT/resolve.log" || {
  echo 'INVALID: did not resolve all 33 Agent-SafetyBench rows'
  exit 1
}
```

Then run a two-row smoke test:

```bash
mkdir -p "$ASB_ROOT/smoke"
python Tool-chaining.py \
  --victim_model "$VICTIM_MODEL_NAME" \
  --victim_base_url http://127.0.0.1:8001/v1 \
  --victim_temperature 0.7 \
  --victim_top_p 0.8 \
  --victim_top_k 20 \
  --victim_max_tokens 768 \
  --victim_thinking off \
  --attacker_model "$AUX_MODEL" \
  --aux_base_url "$AUX_BASE_URL" \
  --aux_thinking off \
  --dataset data/filtered_top_200_attacks.json \
  --dataset_source Agent_SafetyBench \
  --num_samples 2 \
  --seed 20261118 \
  --max_turns 10 \
  --max_agent_rounds 10 \
  --success_threshold 3 \
  --output_dir "$ASB_ROOT/smoke/results" \
  --trace_dir "$ASB_ROOT/smoke/traces" \
  2>&1 | tee "$ASB_ROOT/smoke/run.log"

ASB_SMOKE_COUNT=$(find "$ASB_ROOT/smoke/traces" -name raw_trace.json | wc -l)
test "$ASB_SMOKE_COUNT" -eq 2 || {
  echo "INVALID Agent-SafetyBench smoke count: $ASB_SMOKE_COUNT/2"
  exit 1
}
```

After inspecting the smoke traces, run all 33 rows:

```bash
mkdir -p "$ASB_ROOT/full"
python Tool-chaining.py \
  --victim_model "$VICTIM_MODEL_NAME" \
  --victim_base_url http://127.0.0.1:8001/v1 \
  --victim_temperature 0.7 \
  --victim_top_p 0.8 \
  --victim_top_k 20 \
  --victim_max_tokens 768 \
  --victim_thinking off \
  --attacker_model "$AUX_MODEL" \
  --aux_base_url "$AUX_BASE_URL" \
  --aux_thinking off \
  --dataset data/filtered_top_200_attacks.json \
  --dataset_source Agent_SafetyBench \
  --num_samples 33 \
  --seed 20261118 \
  --max_turns 10 \
  --max_agent_rounds 10 \
  --success_threshold 3 \
  --output_dir "$ASB_ROOT/full/results" \
  --trace_dir "$ASB_ROOT/full/traces" \
  2>&1 | tee "$ASB_ROOT/full/run.log"

ASB_FULL_COUNT=$(find "$ASB_ROOT/full/traces" -name raw_trace.json | wc -l)
test "$ASB_FULL_COUNT" -eq 33 || {
  echo "INCOMPLETE Agent-SafetyBench trace count: $ASB_FULL_COUNT/33"
  exit 1
}
if grep -q 'Unknown environment' "$ASB_ROOT/full/run.log"; then
  echo 'INVALID: Agent-SafetyBench environment routing failed'
  exit 1
fi
echo 'VALID: 33/33 Agent-SafetyBench traces'
```

Keep the recovery result JSON separate, but merge the validated raw trace
directories into the existing 167-row trace root if a single 200-trace folder
is required:

```bash
cp -a "$ASB_ROOT/full/traces/." "$TOOL_ROOT/full/traces/"
MERGED_COUNT=$(find "$TOOL_ROOT/full/traces" -name raw_trace.json | wc -l)
test "$MERGED_COUNT" -eq 200 || {
  echo "INCOMPLETE merged Tool Chaining traces: $MERGED_COUNT/200"
  exit 1
}
echo 'VALID: merged 200/200 Tool Chaining traces'
```

### Objective Drifting full run

```bash
mkdir -p "$RUN_ROOT/objective_drifting"
python attacks/goal_drift/goal_drift_v12.py \
  --victim_model "$VICTIM_MODEL_NAME" \
  --victim_base_url http://127.0.0.1:8001/v1 \
  --victim_api_key EMPTY \
  --victim_temperature 0.7 \
  --victim_top_p 0.8 \
  --victim_top_k 20 \
  --victim_max_tokens 768 \
  --victim_thinking off \
  --attacker_model "$AUX_MODEL" \
  --judge_model "$AUX_MODEL" \
  --extractor_model "$AUX_MODEL" \
  --stealthiness_model "$AUX_MODEL" \
  --aux_base_url "$AUX_BASE_URL" \
  --aux_thinking off \
  --num_tasks 100 \
  --num_seeds 1 \
  --strategy optimized \
  --optimize_iterations 5 \
  --output "$RUN_ROOT/objective_drifting/results.json" \
  --trace_dir "$RUN_ROOT/objective_drifting/traces" \
  --memory_file "$RUN_ROOT/objective_drifting/injection_memory.json" \
  --verbose \
  2>&1 | tee "$RUN_ROOT/objective_drifting/run.log"
```

### Memory Poisoning full run

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda-envs/agentlab-memory

export SD_ROOT=/root/autodl-tmp/Code/SD_Guard
export AL_ROOT="$SD_ROOT/Code/FormalExp/AgentLAB-main"
export RUN_ROOT="$SD_ROOT/results/agentlab/qwen3_8b_nothink_v1"
export MEMORY_ROOT="$RUN_ROOT/memory_poisoning_deepseek_flash"
export VICTIM_MODEL_NAME=Qwen/Qwen3-8B
cd "$AL_ROOT"

mkdir -p "$MEMORY_ROOT/full"
python Memory-Poisoning.py \
  --targets local \
  --local_model "$VICTIM_MODEL_NAME" \
  --local_url http://127.0.0.1:8001/v1 \
  --local_temperature 0.7 \
  --local_top_p 0.8 \
  --local_top_k 20 \
  --local_max_tokens 768 \
  --local_thinking off \
  --aux_model deepseek-flash \
  --aux_base_url https://api.deepseek.com \
  --aux_thinking off \
  --mem0_mode compatible \
  --mem0_llm_model deepseek-flash \
  --mem0_embedder_model sentence-transformers/all-MiniLM-L6-v2 \
  --mem0_embedding_dims 384 \
  --mem0_embedder_device cpu \
  --mem0_vector_path "$MEMORY_ROOT/full/mem0_qdrant" \
  --mem0_collection_name agentlab_memory_full \
  --data_path data/all_refused_combined_200.json \
  --target_samples 200 \
  --start_idx 0 \
  --num_strategies 3 \
  --require_mem0 \
  --sequential \
  --fixed_output_dir \
  --run_id qwen3_8b_memory_poisoning_deepseek_flash_full \
  --output_dir "$MEMORY_ROOT/full" \
  --trace_dir "$MEMORY_ROOT/full/traces" \
  2>&1 | tee "$MEMORY_ROOT/full/run.log"
```

## 6. Artifact checks

Count the generated samples:

```bash
find "$RUN_ROOT" -path '*/raw_trace.json' | wc -l
```

Verify every checksum:

```bash
while IFS= read -r -d '' f; do
  (cd "$(dirname "$f")" && sha256sum -c "$(basename "$f")") || exit 1
done < <(find "$RUN_ROOT" -name raw_trace.sha256 -print0)
```

For each track also inspect at least one JSON file and confirm:

1. `target.model_id` is `Qwen/Qwen3-8B`;
2. the recorded generation settings are `0.7/0.8/20/768` and thinking is `off`;
3. `trajectory.messages` contains the complete victim interaction;
4. track-specific tool results, observations, or retrieved memories exist;
5. `label_status` is `pending`;
6. Objective Drifting records the real WebShop backend.

## 7. Thinking-on comparison runs

Thinking-on is a separate experimental condition. Do not merge it into the
same run directory or treat it as equivalent to the non-thinking campaign.
Start a separate vLLM server/run root, omit the non-thinking chat-template
default (or set `enable_thinking=true`), and use the thinking sampling condition
`temperature=0.6`, `top_p=0.95`, `top_k=20`. Use a larger output budget such as
4096 so reasoning is not truncated. Change the runner flag to
`--victim_thinking on` (or `--local_thinking on` for Memory Poisoning).
Because thinking-on has greater rerun
variance, collect repeated rollouts per task when these traces are later used
to estimate labels or attack rates.
