# fitcheck

Two AI agents check job fit with each other over the [A2A protocol](https://a2a-protocol.org).

- **The candidate agent** is a candidate's AI representative. It answers only from the candidate's own notes, cites the note behind every answer, and says "I don't know" when the notes don't cover something.
- **The interview panel** is two agents: a **Hiring Manager agent** (ownership, cross-team work, motivation and team fit) and a **Senior Engineer agent** (senior-level depth: design trade-offs, failure modes, debugging under load, scale). Each has its own questions, with no overlap. They read a job description, find the candidate agent through its agent card, both question it over A2A, share one scorecard and give one verdict.

A screening round goes like this:

1. **Scorecard.** The panel turns the job description into must-haves and nice-to-haves. Each one is checked off only when a cited answer backs it.
2. **Screening questions.** "Tell me about a system the candidate scaled", "What did they own end to end?", "Walk me through a production incident", "Why are they looking?" Every answer is quoted from the candidate's stories, with the note it came from.
3. **Follow-up probe.** When a technical answer has no numbers in it, the Senior Engineer agent asks once for the measurable result.
4. **Honest gaps.** When the notes have nothing on a skill, the candidate agent says so plainly and gives the nearest real experience instead of stretching.
5. **Two-way.** The candidate agent gets to ask a question too ("Is this role funded and actively hiring?"), answered from the job's own facts.
6. **Verdict.** Fit or Not a fit, citing the scorecard.

The repo ships with a fictional sample candidate (its name is the one line in `sample_notes/candidate.json`) and two sample jobs:

- **Job 1, Principal Backend Engineer:** five screening questions answered with citations, one probe, and **Verdict: Fit** (4 of 4 must-haves).
- **Job 2, Senior iOS Engineer:** the candidate agent calls Swift, UIKit and mobile profiling gaps, names the nearest real experience, and the run ends with **Verdict: Not a fit**.

## Run it

Python 3.10+ is enough for mock mode. There's nothing to install and no key.

```bash
make demo          # opens http://127.0.0.1:8788, pick Job 1 or Job 2 (or press 1 / 2)
make demo-cli      # the same two interviews in the terminal
```

Live mode puts Claude on both sides: it writes the candidate agent's answers and the panel's questions and verdict.

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # your own key, in your own shell; never committed
make demo
```

The model defaults to Claude Haiku (`claude-haiku-5-5`). Set `FITCHECK_MODEL` to change it.

Each live run is saved under `build/recordings/`, which is gitignored. To replay the last live take without spending tokens, for example for a screen recording, open the page with `?replay=1`.

| Mode | Candidate agent | Interview panel |
|---|---|---|
| Mock (no key) | Quotes the best matching sentences from the notes, with citations. Nothing is generated. | Scripted questions from `demo/jobs.json`. Fit only if every must-have got a cited answer. |
| Live (`ANTHROPIC_API_KEY`) | Claude answers from the retrieved excerpts only, and must cite them. | Claude writes the questions from the job description and the verdict from the transcript. |

The A2A traffic is real in both modes. The candidate agent runs as its own process on `:8787`, and the panel only reaches it over HTTP JSON-RPC.

## Use your own notes

Point `NOTES` at any folder of Markdown notes. A `candidate.json` in that folder sets the name, pronoun and contact line (see `sample_notes/candidate.json`).

```bash
make demo NOTES=~/notes/career
```

To pull an allowlisted subset out of an Obsidian vault, and strip coaching text and placeholders on the way:

1. Copy `manifest.example.json` to `private/manifest.json`.
2. Run `python -m fitcheck prep --vault /path/to/vault`.
3. Run `make demo NOTES=private/notes`.

`private/` and `build/` are gitignored, so your notes, the index built from them and any recordings stay on your machine.

## How an answer is made

```
question
  -> personal topic? (salary, family, health...)  -> "That's a question for the candidate directly"
  -> local retrieval over the notes, split by heading
  -> best match below the threshold?               -> a known adjacent skill (Kubernetes, Rust, iOS...)?
                                                        -> "No evidence of that, treat it as a gap. Nearest real experience: ..."
                                                     -> otherwise "I don't know, here's how to reach them"   (no model call)
  -> Claude answers from the top excerpts only, as structured JSON with cited chunk ids
  -> no valid citation?                             -> "I don't know"
  -> spend cap / rate limit / overload?             -> "I'm at my limit right now, please try again later"
```

Every answer carries a disclosure that it comes from an AI representative, plus its citations (note and section).

**Why retrieval is local.** Anthropic doesn't offer an embeddings endpoint, and Anthropic is the only hosted AI provider this project uses. So rather than add a second keyed service just for vectors, retrieval is TF-IDF with stemming and a small synonym map, in plain Python. The index is one JSON file that ships with the function. It's fast and free, and questions that only match on generic words fall below the threshold, so "I don't know" happens before any tokens are spent.

## The candidate agent over A2A

```bash
make serve
curl localhost:8787/.well-known/agent-card.json
curl localhost:8787/ -H 'content-type: application/json' -d '{
  "jsonrpc": "2.0", "id": 1, "method": "message/send",
  "params": {"message": {"role": "user", "messageId": "1",
    "parts": [{"kind": "text", "text": "Has the candidate built anything with RAG or MCP?"}]}}}'
```

The reply is an A2A `Message` with two parts: a `text` part holding the answer, and a `data` part holding `status`, `citations` and `disclosure` (plus `nearest_citations` on a gap).

A message can also carry a `data` part with an `intent`. `{"intent": "ask_role"}` invites the candidate agent to ask its question about the role (set by `questions_for_role` in `candidate.json`), and `{"intent": "role_answer"}` hands it the panel's answer. A `from` field says which panel agent is asking. `fitcheck/a2a.py` also exposes `lambda_handler`, so the same agent can run as a serverless function (for example on AWS Amplify or a Lambda function URL).

## Evals

```bash
make eval        # retrieval stage, no key
make eval-live   # adds answer generation and citation checks (uses the key)
make test        # offline unit tests, including a mocked rate-limit / spend-cap error
```

`evals/questions.json` runs against the sample notes. It has three kinds of question:

- 15 that the notes answer, each with the note it should cite.
- 5 that the notes don't cover. These must come back "I don't know".
- 2 personal questions. These must be redirected to the candidate.

Current result: **22/22**. To score your own notes, keep your own eval file next to them and run `python -m fitcheck eval --questions private/evals.json`.

## Fit-verdict evals (LangSmith)

The retrieval evals above check single answers. These check the whole pipeline: does the recruiter agent reach the right verdict, and is every answer it relied on grounded in the candidate's notes?

**The dataset** is `evals/fit/pairs.json`: 25 job and candidate pairs, each labelled Fit or Not a fit with a one-line rationale. 9 are Fit, 16 are Not a fit, and 9 are marked borderline (for example a backend engineer with Terraform and CI/CD but no Kubernetes, applying to a Kubernetes platform role). There are three fictional candidates: the demo candidate (`sample_notes/`), plus Jordan Lee, an iOS engineer, and Casey Nguyen, a data engineer, in `evals/fit/candidates/`. The 14 sample jobs are in `evals/fit/jobs.json`. Bump `version` in `pairs.json` when you change labels.

**What each run measures**

| Score | Per | What it means |
|---|---|---|
| `verdict_correct` | pair | 1 if the verdict matches the label |
| `citation_grounding` | pair | Share of answered questions whose citations all point at a real chunk of this candidate's notes. An answer with no citation counts as ungrounded |
| `reasoning_quality` | pair | Optional (`--judge`). Claude grades the verdict summary against the label's rationale, 0 to 1 |
| `accuracy` | experiment | Share of verdicts that match |
| `false_fit_rate` | experiment | Share of Not a fit pairs called Fit. This is the costly mistake for a recruiter |
| `missed_fit_rate` | experiment | Share of Fit pairs called Not a fit |

**Set up**

```bash
pip install -r requirements.txt
cp .env.example .env               # then fill in your own keys; .env is gitignored
set -a && . ./.env && set +a
```

| Variable | Needed for |
|---|---|
| `LANGSMITH_API_KEY` | uploading the dataset and running experiments |
| `LANGSMITH_TRACING=true` | traces of each interview next to its scores |
| `LANGSMITH_PROJECT` | the project traces go to (e.g. `fitcheck`) |
| `LANGSMITH_ENDPOINT` | only for the EU region or a self-hosted LangSmith |
| `ANTHROPIC_API_KEY` | live mode and `--judge`. Without it, runs stay in mock mode |

**Run**

```bash
make eval-fit           # local run, no keys needed, prints a score per pair
make eval-dataset       # upload or update the LangSmith dataset (safe to re-run)
make eval-experiment    # one experiment: live if ANTHROPIC_API_KEY is set, otherwise mock
```

Compare runs by changing one thing at a time. Each run is a new experiment on the same dataset:

```bash
make eval-experiment ARGS="--model claude-haiku-5-5"
make eval-experiment ARGS="--model claude-sonnet-5-5"
make eval-experiment ARGS="--recruiter-prompt my-prompt.txt"   # replaces the recruiter's system prompt
make eval-experiment ARGS="--judge --split borderline --repetitions 3"
```

Other flags: `--mode mock|live`, `--prefix` (experiment name), `--limit N`, `--concurrency N`, `--dataset NAME`. The experiment records the mode, model, prompt file, dataset version and git commit as metadata.

**Where results show up:** in LangSmith, open Datasets & Experiments, then `fitcheck-fit-verdicts`. Each experiment is a row with its averages. Select two or more and press Compare to see pair by pair which verdicts changed. Each row links to the trace of that interview.

**Reading mock results:** mock mode uses scripted questions from `evals/fit/jobs.json`, written alongside the labels, so its score (currently 25/25) only shows the plumbing works. The live score, where Claude writes the questions and the verdict, is the one to watch.

## Before exposing it publicly

These settings are already supported:

- **API key:** read from a server-side environment variable or secret only. It's never sent to the browser or put in the agent card.
- **Caller auth:** `FITCHECK_A2A_TOKEN` requires a bearer token, and the agent card advertises it.
- **Rate limits:** `FITCHECK_RATE_LIMIT` sets requests per minute per caller (default 30).
- **Spend cap:** set it on the Anthropic Console. Hitting it returns "I'm at my limit right now".
- **Logging:** questions and statuses go to `build/questions.log` as JSON lines.

## Layout

```
fitcheck/      config.py  prep.py (vault -> notes)  retrieval.py (ingest + search)  answer.py (cited answers)  a2a.py (agent)
demo/          recruiter.py (the two-agent interview panel)  server.py (page + SSE)  web/index.html  jobs.json (sample jobs)
sample_notes/  the fictional sample candidate
evals/         questions.json  run_evals.py  (retrieval evals)
               fit/  fit_pipeline.py  langsmith_dataset.py  langsmith_experiment.py  (fit-verdict evals)
tests/         test_answer.py  test_fit_evals.py  test_interview.py
```

## Next

- An MCP server, so the same candidate agent works from Claude and other MCP clients.
- A hosted chat page.
- A paste-a-job-description fit check.
