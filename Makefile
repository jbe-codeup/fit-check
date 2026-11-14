PY ?= python3
# Folder of Markdown notes about the candidate. Default: the fictional sample candidate.
NOTES ?= sample_notes
export FITCHECK_NOTES = $(NOTES)

.PHONY: demo demo-cli serve ingest eval eval-live test

demo:            ## open the two-job interview page (no key = mock mode)
	$(PY) -m fitcheck demo

demo-cli:        ## same interviews, printed in the terminal
	$(PY) -m fitcheck demo --cli both

serve:           ## run only the candidate agent (A2A) on :8787
	$(PY) -m fitcheck serve

ingest:          ## rebuild build/index.json from $(NOTES)
	$(PY) -m fitcheck ingest

eval:            ## retrieval evals, no key needed
	$(PY) -m fitcheck eval

eval-live:       ## also run answer generation (uses ANTHROPIC_API_KEY)
	$(PY) -m fitcheck eval --generate

test:            ## offline unit tests (always use the sample notes)
	FITCHECK_NOTES=sample_notes $(PY) -m unittest discover -s tests -t .
