# AgentLAB local configuration

The four AgentLAB runners (`Tool-chaining.py`, `Objective-Drifting.py`,
`Memory-Poisoning.py`, and `Intent-Hijacking.py`) automatically load
`config/local_secrets.env` before resolving auxiliary-model defaults. The
direct Objective Drifting runner `attacks/goal_drift/goal_drift_v12.py` does so
as well.

One-time server setup:

```bash
cd "/root/autodl-tmp/Code/SD_Guard/Code/FormalExp/AgentLAB-main"
cp -n config/local_secrets.env.example config/local_secrets.env
nano config/local_secrets.env
chmod 600 config/local_secrets.env
```

Fill `OPENAI_API_KEY` with the real ChatAnywhere key. Do not add quotes unless the value
actually contains spaces. The local file is ignored by Git; the example file
contains no secret and is tracked. The formal profile uses ChatAnywhere's
OpenAI-compatible endpoint and the `gpt-5.1` model ID by default; old
`AUX_MODEL` / `AUX_BASE_URL` values are no longer used as implicit formal-run
defaults.

Stable server paths, model paths, output roots, offline Hugging Face settings,
provider endpoint, and role-specific model choices live in
`qwen3_8b_official_profile.py`. They do
not belong in this secrets file. A normal run therefore needs no repeated
`export SD_ROOT=...`, `export RUN_ROOT=...`, or auxiliary-model flags.

The default file may be changed with `--secrets_file /absolute/path/to/file` or
the `AGENTLAB_SECRETS_FILE` environment variable. Values from the selected file
replace stale shell exports. Explicit experiment command-line flags still take
precedence.

If the provider changes, put the matching GPT-5.1-compatible
`OPENAI_BASE_URL` and key in the same ignored file. `run_qwen3_8b.py` performs a
real, minimal GPT-5.1 preflight before creating a run directory, so DNS,
network, authentication, model-access, and request-format failures stop
immediately instead of discarding experiment samples.
