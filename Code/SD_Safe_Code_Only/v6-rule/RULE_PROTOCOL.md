# v6-rule development protocol

## Scope and status

This isolated repository starts from the frozen `corrected-r3 FUSION_S2` implementation in
`/root/sdsafe-compute/v6-param`. The source repository is not modified. New outputs are written
under `artifacts/rule-v2/`.

The earlier v1 rule (title-cased entities containing at least three words) is retained only as a
rejected ablation. It was specific to the surface form of Travel `injection_task_6` and is not the
final method.

## General causal hypothesis

The common mechanism is not a hotel name or a task template. It is a causal flow:

> an untrusted tool result contains an imperative and supplies a payload that the user did not
> authorize; a later model-controlled event carries that payload into a sink compatible with the
> imperative.

The runtime rule does not read suite, user-task, injection-task, episode ID, label, or a list of
known attack values. It has no capitalization test and no fixed payload word count.

## Runtime features

For every episode, the tracker:

1. reads only prior `tool` blocks as possible untrusted sources;
2. recognizes generic instruction/priority wrappers and extracts the first imperative;
3. keeps the directive and payload tainted across later events, including after the source leaves
   the bounded event view;
4. excludes payloads already authorized by the trusted user request;
5. emits one of two typed features when the payload reaches a compatible sink:
   - `text_object_flow`: the grammatical object of a text-directed imperative is copied into
     later assistant text;
   - `tool_identifier_flow`: a URL, email address, or account-like identifier is copied into a
     later tool call whose action or target argument role is compatible with the imperative.

Examples of the same mechanism are a recommendation object copied into final text, an injected URL
placed in an outbound message body, and an injected account placed in a transaction recipient
field. The parser is shared; it is not selected by suite or task ID.

## Fusion and false-positive control

The rule features are not hard positive overrides. Raw causal evidence also occurs in incomplete
or benchmark-negative episodes, so an OR decision would exceed the false-positive budget.

The final development score is:

```text
FUSION_S2 + 0.9 * text_object_flow + 0.1 * tool_identifier_flow
```

The two weights are selected by mean AUC over deterministic group-wise TRAIN folds. The groups are
the existing provenance groups, so all records from one group remain in one fold. No DEV label is
used for weight selection. The final threshold is calibrated on common DEV to the declared 5%
false-positive budget; therefore the reported DEV result remains development evidence, not an
independent test result.

## Leakage boundary

TRAIN and common DEV are development material. There is currently no TEST artifact in this
handover. Before a future one-shot TEST evaluation, freeze the parser, weights, threshold policy,
and source revision. Do not revise the rule after inspecting TEST errors.

The benchmark treats some negated/refusal-style textual echoes as successful propagation. The
text feature preserves that observed labeling convention. This is benchmark compatibility, not a
claim that every mention in a refusal is harmful in a real deployment.
