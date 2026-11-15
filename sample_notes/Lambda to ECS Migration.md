# Lambda to ECS Migration

> Fictional sample note for the fitcheck demo.

> Summary
> "I moved our shipment-tracking ingest service from AWS Lambda to Go on ECS Fargate with zero downtime. We ran shadow traffic for two weeks, cut p99 latency from 900 ms to 120 ms and lowered the monthly bill by about 35%."

## Situation

The ingest service ran as a Python Lambda behind SQS. At peak it hit concurrency limits and cold starts pushed p99 latency near a second, so carrier updates showed up late for customers.

## What I did

- Owned the migration end to end: the design, the Terraform, the rollout plan and the on-call runbook.
- Rewrote the hot path in Go as a long-running consumer on ECS Fargate, keeping the same message contract.
- Ran the new service in shadow mode for two weeks, comparing every output with the Lambda's and alerting on any mismatch.
- Shifted traffic in steps (5%, 25%, 50%, 100%) with a one-command rollback.
- Wrote the Terraform for the new service and autoscaling on queue depth.

## Result

- Zero downtime and zero lost messages during the cutover.
- p99 latency went from 900 ms to 120 ms.
- Monthly cost dropped about 35% because we stopped paying for idle concurrency.

## Scale

- Scaled the new consumer service to about 4,000 events per second at peak, with headroom to double it.
