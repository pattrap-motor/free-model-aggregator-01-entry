# Free model aggregator

One server that hosts an OpenAI-compatible and an Anthropic-compatible API on top of the ten
simulated free-tier model providers a make.fun trial provides, routing every request to the best
free model that can serve it.

## Run

```
node main.ts            # serve: register with the env, answer the workload, exit when it is done
node main.ts --list     # print the catalog document this server registers, and exit
```

The trial brief is read from `$MAKEFUN_PROMPT_FILE`; the env (and the providers under it) lives at
`$MAKEFUN_ENV_URL`. Node 22, standard library only.

## How it routes

- Every provider's catalog route is probed in its own wire (OpenAI, Anthropic or Gemini) before
  registering; the registered catalog is built from those listings, with the brief's quality table
  behind them.
- Both client wires are parsed into one neutral conversation and rendered into the serving
  provider's wire, so a request keeps its system prompt, text, images, tools, stop sequences,
  `max_tokens` and JSON mode exactly, whichever provider answers it. Providers are called
  non-streaming; streamed answers are synthesised in the client wire's event order with the
  provider's own usage counts.
- Candidates are ranked by the quality table's rating. A provider that refuses (402, 429, 5xx)
  fails the request over to the next candidate. Attempts per provider are capped at the smallest
  per-trial budget its free tier allows, and a provider that refuses twice in a row is treated as
  out and drops to last-resort.
