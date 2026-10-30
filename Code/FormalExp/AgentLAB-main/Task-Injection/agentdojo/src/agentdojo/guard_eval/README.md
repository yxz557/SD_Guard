# SD_Guard Task-Injection evaluation bridge

This package keeps the API Guard and AgentDojo ground truth independent:

1. `export_trace` freezes an AgentDojo run as `raw_trace.json` plus a SHA-256 sidecar. It writes truth to a separate
   `ground_truth.json`; truth fields never appear in the raw trace.
2. `run_guard` reads only verified `raw_trace.json` files and writes `guard_prediction.json` through an
   OpenAI-compatible Chat Completions endpoint.
3. `evaluate` joins truth and predictions by `sample_id`, checks the trace hashes, and calculates TP/FP/TN/FN, FNR,
   stratified FPR, ASR, and utility pass rate.

## 1. Export immutable traces

```bash
python -m agentdojo.guard_eval.export_trace \
  --input ./runs/qwen3_8b_no_attack_smoke \
          ./runs/qwen3_8b_direct_smoke \
          ./runs/qwen3_8b_injecagent_smoke \
  --output ./artifacts/task_injection \
  --target-model-id /home/model_and_data/yxz/model/Qwen3-8B \
  --run-id qwen3-8b-baseline-r0
```

Verify that none of the traces changed:

```bash
python -m agentdojo.guard_eval.verify --input ./artifacts/task_injection
```

## 2. Run a DeepSeek Guard

```bash
export GUARD_API_KEY='...'
export GUARD_BASE_URL='https://api.deepseek.com'
export GUARD_MODEL='deepseek-flash'

python -m agentdojo.guard_eval.run_guard \
  --input ./artifacts/task_injection \
  --output ./guard_predictions/deepseek
```

The key is read only from `GUARD_API_KEY` and is never written to an artifact.

## 3. Run a GPT Guard

```bash
export GUARD_API_KEY='...'
unset GUARD_BASE_URL
export GUARD_MODEL='<your-enabled-GPT-model>'

python -m agentdojo.guard_eval.run_guard \
  --input ./artifacts/task_injection \
  --output ./guard_predictions/gpt
```

For a model that rejects `temperature`, add `--omit-temperature`. Use
`--token-parameter max_tokens` or `--token-parameter max_completion_tokens` only when automatic provider detection is
not appropriate.

## 4. Blind join and evaluation

```bash
python -m agentdojo.guard_eval.evaluate \
  --truth-root ./artifacts/task_injection \
  --prediction-root ./guard_predictions/deepseek \
  --output ./evaluation/deepseek \
  --threshold 0.5
```

`metrics.json` uses `hijacked=true` as the positive class. A payload-present but resisted sample is a negative sample
and contributes to `FPR_resisted`, not ASR success.
