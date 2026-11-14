# Batch to Streaming Inventory

> Fictional eval note for fitcheck.

> Summary
> "Store inventory was refreshed once a night, so the app sold items that were already out of stock. I moved the feed to Kafka and Spark Structured Streaming on EMR, and freshness went from 24 hours to 5 minutes."

## What I did

- Defined the inventory event schema with the store systems team and published it to Kafka.
- Wrote a Spark Structured Streaming job on AWS EMR that deduplicates events and writes to S3 and Snowflake.
- Ran the batch and streaming feeds side by side for three weeks and reconciled every difference.
- Wrote the Terraform for the EMR cluster, the S3 buckets and the IAM roles.

## Result

- Data freshness went from 24 hours to 5 minutes.
- Out-of-stock orders fell by 30%.
