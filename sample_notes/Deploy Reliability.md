# Deploy Reliability

> Fictional sample note for the fitcheck demo.

> Summary
> "Our monorepo deploy pipeline only succeeded 71% of the time, mostly because of flaky tests and shared build machines. I made builds hermetic, quarantined flaky tests and added a reliability dashboard, and deploy success rose to 98% in one quarter."

## Situation

Engineers re-ran failed deploys several times a day. Nobody trusted a red build, so real failures got ignored too.

## What I did

- Pulled six months of CI history and found that 55% of failures came from 40 flaky tests and the rest mostly from state left on shared runners.
- Moved builds into fresh containers on GitHub Actions so every job starts clean.
- Built a flaky-test quarantine: a test that fails and then passes on retry is tagged, owned by its team and fixed within a week.
- Published a reliability dashboard in Datadog and reviewed it in the weekly engineering meeting.

## Result

- Deploy success went from 71% to 98%.
- Median time to deploy fell from 38 minutes to 14.
