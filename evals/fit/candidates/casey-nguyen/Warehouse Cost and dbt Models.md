# Warehouse Cost and dbt Models

> Fictional eval note for fitcheck.

> Summary
> "Our Snowflake bill doubled in a year because every dashboard rebuilt full tables. I rebuilt the core models in dbt as incremental models with tests, and Snowflake spend fell by 40% while dashboards got faster."

## What I did

- Mapped which dashboards and Airflow DAGs read which tables, and found 60 tables nobody used.
- Rewrote the order, customer and inventory models in dbt as incremental models with a star schema.
- Added dbt tests for uniqueness, nulls and freshness, and alerts to the data on-call channel.
- Ran weekly office hours with analysts while they moved their dashboards to the new models.

## Result

- Snowflake spend fell by 40%.
- Median dashboard load time went from 40 seconds to 6.
