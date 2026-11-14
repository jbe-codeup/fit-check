# Offline Trail Maps with Core Data

> Fictional eval note for fitcheck.

> Summary
> "Hikers lost their maps whenever signal dropped. I rebuilt offline trail maps on Core Data with background sync and conflict handling, and offline-related support tickets fell by 70% in two months."

## Situation

The app cached map tiles in an ad hoc folder and saved trail edits only when online, so edits made on the trail were lost.

## What I did

- Modeled trails, waypoints and pending edits in Core Data, with a lightweight migration from the old cache.
- Used background URLSession downloads for tile packs and BGTaskScheduler for sync.
- Wrote a last-writer-wins merge with a manual resolve screen for the rare real conflict.
- Covered the sync engine with XCTest, including airplane-mode and app-killed cases.

## Result

- Offline-related support tickets fell by 70%.
- Zero data-loss reports in the following six months.
