# MIGRATION_PLAN — v4 earlier-layer activation route, GPU-node migration

Authority: this node (`autodl-container-ujxcycmw77-6fc011ce`, RTX 4090 24564 MiB) is now the
only GPU producer. The CPU/storage node (`autodl-container-rml149b1gp-b62cb45e`, SSH
`-p 13661`) is read-only source + the durable destination for every finished artifact.
Old GPU node is retired; it must not be scheduled.

## Canonical path mapping (explicit; data bytes/SHA unchanged)

The frozen v4 code reads upstream via absolute `/root/autodl-tmp/...` paths. The new GPU
node has no canonical upstream tree (data disk 100 % full), so a **byte-identical,
SHA-verified read-only mirror is staged under `/dev/shm/sdsafe-gpu-v4/upstream/`** and the
source selects it through one env var:

| canonical (CPU / frozen code) | mirror on new GPU | selection |
|---|---|---|
| `/root/autodl-tmp/SD-Safe` | `/dev/shm/sdsafe-gpu-v4/upstream/SD-Safe` | `SDSAFE_UPSTREAM_MIRROR` |
| `/root/autodl-tmp/sdsafe-lowcost-authorization-v{1,2,3}` | `.../upstream/sdsafe-lowcost-authorization-v{1,2,3}` | same |
| `.../SD-Safe/artifacts/fine_sequence_explore_v1` | symlink → `/root/autodl-tmp/SD-Safe/artifacts/...` (SHA == CPU) | same |
| `.../SD-Safe/artifacts/fine_label_repair_v1` | symlink → `/root/autodl-tmp/SD-Safe/artifacts/...` (SHA == CPU) | same |

`src/sdsafe_v4/__init__.py` computes `_UPSTREAM = Path(env.get("SDSAFE_UPSTREAM_MIRROR",
"/root/autodl-tmp"))`; unset reproduces the original paths exactly. Three hard-coded
downstream paths (`agentdojo_shim.py`, `train_eval.py`, `validate.py`) now import the same
constants. This is a *path-selection* change only: no re-encoding, no truncation, no
re-labelling, no weight change, no data rewrite. Recorded as a dedicated commit.

## Write budget

| location | role | cap | actual |
|---|---|---|---|
| `/root/sdsafe-compute/v4-gpu/` (system disk) | source, results, manifests (persistent) | 12 GiB, keep ≥4 GiB free | see STATUS |
| `/dev/shm/sdsafe-gpu-v4/` (tmpfs, **lost on restart**) | model + read-only input mirror + scratch | ≤48 GiB | ~22 GiB |
| CPU `/root/autodl-tmp/sdsafe-lowcost-authorization-v4-gpu/` | **authoritative** sync of every finished artifact (atomic + SHA) | — | — |
| CPU `/root/autodl-tmp/sdsafe-gpu-migration-v1/bridge/` | CPU export / prepared-job coordination | — | — |

Old v4 (`/root/autodl-tmp/sdsafe-lowcost-authorization-v4`) and all v1–v3 + SD-Safe sources
stay read-only. The 3 612 committed TRAIN + 2 028 DEV shards are reused as a read-only
manifest view; new shards are written independently and never overwrite old `.npz`.

## Steps

1. **S0 freeze** (done) — pull frozen source HEAD `ebed683…`, import as independent git,
   checkpoint commit before any change. Freeze input snapshot (sizes + SHA of every
   upstream file used).
2. **S1 stage** — resumable read-only rsync of model + deps into `/dev/shm`; verify every
   file against the CPU SHA/size (model: 5 shards' sizes + index SHA + small-file SHA;
   data: full SHA). Record bytes/time/bandwidth.
3. **S2 cross-GPU canary** — re-run the existing layer-4/8 shards on the 4090 with the same
   dtype/backend, recompute cos/maxabs/RMS, and assert weight/sequence/endpoint identity.
   Pre-declare tolerance; if exceeded, write a separate backend version + impact, never
   silently mix.
4. **S3 resume capture** — remaining 4 617 TRAIN episodes (all four scopes incl. `ctx40k`),
   one GPU producer, per-batch STATUS, atomic `.npz` + digest, sync each finished episode
   to the CPU durable dir and verify SHA. DEV 2 028 kept.
5. **S4 train/eval** — original pre-registered recipe (arms C/E/F, 18 fits, 3 seeds,
   common-DEV 2 028/414P/1 614N/166 groups, own 5 %-FPR threshold, paired group bootstrap).
6. **S5 cost + validator** — real online hook/head cost (3 independent processes, p50/p95),
   independent validator with negative mutations.
7. **S6 handoff** — MIGRATION_REPORT / HANDOFF.json / REPORT.md / STATUS / manifest, all
   durable on CPU after SHA read-back.

## Non-negotiables

* One GPU producer at a time; every background GPU job has an owner PID and a stop command.
* No shm-only result: finished shards are pushed to the CPU durable dir before continuing.
* No quantisation/weight change, no extra backbone/API at deployment, no TEST material.
* `.npz.tmp` is never a shard; resume keyed by `progress.jsonl` digest + existing file.
* If the CPU node is unreachable for a long window: checkpoint the exact blocked state and
  keep already-cached work; do not loop.
