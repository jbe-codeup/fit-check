# Production Incident: Carrier Webhooks

> Fictional sample note for the fitcheck demo.

> Summary
> "When a carrier changed their webhook signature format without notice, we silently dropped tracking updates for 40 minutes. I was incident commander: I found the cause, shipped a fix that accepted both formats, replayed the 180,000 dropped events from the dead-letter queue, and wrote the postmortem."

## Situation

A large carrier rotated how they signed webhooks on a Friday evening. Our verifier rejected every request, and because rejections were logged at debug level, no alert fired. Customers noticed before we did.

## What I did

- Took incident command, opened a status page note within 10 minutes of the first customer report, and pulled in the carrier's on-call.
- Found the cause by diffing a rejected payload against a good one, then shipped a verifier that accepted both signature formats.
- Replayed the 180,000 events from the dead-letter queue in order, with a dry run first.
- Ran the blameless postmortem and owned its three follow-ups.

## Result

- All 180,000 dropped events were recovered, with no duplicate updates.
- Rejected webhooks now page on-call within 2 minutes instead of failing silently.
- The follow-up contract test with carriers caught two more format changes before they reached production.
