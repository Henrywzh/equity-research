# LLM Provider Refactor

## Runtime Shape

`daily_macro.analysis` creates a `ModelPool` from configured `ProviderAccount`
objects. Each account has a provider, credential identity, endpoint URL, and
explicit `quota_scope`. A `ModelConfig` adds the model capability and derives a
stable endpoint id:

```text
provider:account:model
```

The resolver scores these endpoints per task. It rejects deprecated/inactive
models, unsupported JSON, context overflow, output overflow, exhausted daily
budgets, and waits longer than the configured wait cap. A production model is
preferred unless a task explicitly allows previews.

## Provider Lanes

- Groq: existing multi-key compatibility. The active allowlist is `qwen/qwen3.6-27b`, followed by `openai/gpt-oss-120b` and `openai/gpt-oss-20b`. Older Llama and Qwen 3.0/3.2 models cannot enter the runnable pool. Keys default to one organization quota scope so they do not falsely multiply organization limits.
- Cerebras: each configured key is an account endpoint by default. The current catalog includes `gpt-oss-120b`, `gemma-4-31b`, and `zai-glm-4.7` with the supplied 65,536/8,192 context limits and quota figures.
- Google AI Studio: Gemini Flash-Lite is the bulk extraction/repair candidate; Gemini Flash is a higher-quality synthesis candidate. Exact project limits are learned from responses and should be checked in AI Studio.
- OpenRouter: free model variants are optional overflow lanes. Their model catalog should be refreshed because free variants can expire or change upstream.

Live non-Groq catalogs are allowlisted rather than copied wholesale. This is
important for OpenRouter, whose catalog can contain hundreds of models that are
not part of the daily-macro policy. Add a model explicitly through the provider
model environment override before it can enter the resolver.

## Quota Accounting

`RateLimitGovernor` and `DailyBudgetLedger` use `quota_scope:model_id` for
shared organization/project limits. Response headers are normalized for both
Groq-style headers and Cerebras minute/day headers. Reports include raw task
counts plus endpoint-level task and token usage, which makes provider selection
auditable without exposing credentials.

## Recommended Initial Policy

1. Use Gemini Flash-Lite or Cerebras Gemma for high-volume article extraction.
2. Use Cerebras/Groq GPT-OSS 120B or Gemini Flash for category synthesis, top alerts, and critic checks.
3. Keep OpenRouter free variants as overflow only until their observed JSON and citation quality is measured.
4. Keep Groq's organization quota scope shared unless the credentials truly belong to independent organizations and that use is permitted by provider terms.

The three pasted Cerebras keys were intentionally not added to the repository;
they should be revoked and replaced locally before enabling these lanes.
