# Serverless Tracking Pipeline on Lambda

> Fictional sample note for the fitcheck demo.

> Summary
> "I rebuilt our shipment-tracking ingest pipeline as Go functions on AWS Lambda, fed by SQS and EventBridge, with state in DynamoDB. We cut over with zero downtime, cut p99 latency from 900 ms to 120 ms and lowered the monthly bill by about 35%."

## Situation

The old ingest path was one large Python Lambda behind SQS. At peak it hit the account's concurrency limit and cold starts pushed p99 latency near a second, so carrier updates showed up late for customers.

## What I did

- Owned the rebuild end to end: the design, the Terraform, the rollout plan and the on-call runbook.
- Split the monolith function into small Go handlers on arm64, which brought cold starts under 100 ms, and gave each one its own SQS queue and reserved concurrency, so one noisy carrier couldn't starve the rest.
- Made every handler idempotent: it records the event ID with a DynamoDB conditional write, so a retried message is acknowledged and skipped instead of applied twice.
- Turned on partial batch responses, so one bad message goes to the dead-letter queue instead of failing the whole batch.
- Added structured logs, CloudWatch metrics per carrier and X-Ray traces, with alarms on queue age and DLQ depth.
- Shifted traffic with weighted Lambda aliases (5%, 25%, 50%, 100%) and a one-command rollback.

## Result

- Zero downtime and zero lost messages during the cutover.
- p99 latency went from 900 ms to 120 ms.
- Duplicate updates from retries went from about 200 a day to zero.
- Monthly cost dropped about 35% after right-sizing memory and removing idle provisioned concurrency.

## Scale

- Scaled the pipeline to about 4,000 events per second at peak, with headroom to double it.
