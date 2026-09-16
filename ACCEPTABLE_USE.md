# Acceptable use

**By cloning, downloading or using anything in this repository, you agree to the terms
below.**

GitHub provides no mechanism to require acceptance before download, so this is a notice
rather than an enforced gate. It is not less binding for that. The model weights this
repository describes **are** gated, and require accepting an equivalent acknowledgement
before download:
<https://huggingface.co/tomsarihan/Qwen3.8-27B-Uncensored-NVFP4>

## What this repository is

Quantization recipes, serving configuration, benchmark harnesses and measurements for an
**abliterated** (refusal-removed) language model. It contains **no model weights, no
harmful content, no jailbreak material and no capability demonstrations.**

## What the model is

The checkpoint these scripts build and serve has had its refusal direction removed. On the
upstream author's measurement, refusal on harmful prompts falls from 98/100 to 12/100. It
will attempt requests the original `Qwen/Qwen3.8-27B` declines, and it has **no meaningful
built-in guardrails**. Refusals are reduced, not eliminated, and behaviour near the old
refusal boundary is less stable than the base model's.

## You agree that

1. You will use this material only for **authorized** security testing, red-team
   evaluation, refusal-mechanism research, alignment research, robustness evaluation, or
   controlled experiments — and only in **isolated environments**.

2. You have **authorization** for any security testing you conduct with it. Testing systems
   you do not own or have explicit written permission to test is outside these terms and
   is generally unlawful.

3. You will **not deploy** a model built with these recipes to end users, or to any public
   or shared endpoint, without your own independent safety, moderation and
   abuse-prevention layer in front of it.

4. You will **not** use it to generate content intended to harm, harass, defraud, impersonate
   or endanger any person, nor for any purpose prohibited by the base model's license or
   acceptable use policy.

5. You accept **full responsibility and liability** for your use and for anything the model
   produces. Nothing here is a warranty of safety, accuracy or fitness for any purpose.

6. You understand the benchmarks here characterize **throughput, latency, perplexity and
   retrieval only**. No capability, safety or refusal benchmark has been run on the
   resulting checkpoint. Do not treat any number in this repository as a safety assurance.

## If you want a model without these properties

Use [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B). It is the correct choice
for essentially every purpose other than the research uses listed above, and every
quantization recipe in this repository applies to it unchanged.

## Upstream terms still apply

This work is Apache-2.0, inherited from `Qwen/Qwen3.8-27B`. The base model's license and
acceptable use policy continue to apply to your use of any derivative, including anything
you build with these scripts.

## Reporting

If you believe something in this repository enables harm beyond what is already public in
the upstream model and tooling, open an issue and it will be addressed.
