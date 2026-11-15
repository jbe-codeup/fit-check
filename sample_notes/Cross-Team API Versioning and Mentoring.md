# Cross-Team API Versioning and Mentoring

> Fictional sample note for the fitcheck demo.

> Summary
> "I led a project across three teams (tracking, billing and integrations) to version our public tracking API without breaking customers, and I mentored two junior engineers through it with weekly pairing. Both shipped major pieces, and one was promoted within the year."

## Situation

Three teams changed the tracking API independently, and two customer integrations broke in one quarter.

## What I did

- Wrote the versioning proposal and got sign-off from the three team leads and the support lead.
- Set up contract tests that every team runs before merging an API change.
- Paired weekly with two junior engineers, gave each a real slice to own (the contract test runner and the deprecation emails) and reviewed their designs before code.

## Result

- No customer-facing API breaks in the following year.
- Both junior engineers shipped their pieces; one was promoted to mid-level.
