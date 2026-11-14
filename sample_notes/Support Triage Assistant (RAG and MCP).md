# Support Triage Assistant (RAG and MCP)

> Fictional sample note for the fitcheck demo.

> Summary
> "I built an internal assistant that drafts triage notes for support tickets using RAG over our runbooks, plus an MCP server that lets our team's AI tools search internal docs. An eval set of 120 labeled tickets kept it honest, and first-response time dropped by a third."

## What I did

- Indexed about 250 runbooks and postmortems, split by heading, with citations back to each source page.
- Built a small MCP server exposing read-only doc search, so the team's assistants could use the same index.
- Wrote an eval set of 120 labeled tickets, including ones the docs can't answer, and ran it on every prompt or retrieval change.
- Made the assistant say "no runbook covers this" instead of guessing.

## Result

- First-response time on support tickets dropped by about a third.
- Wrong suggestions fell from 18% to 4% across three eval iterations.
