# App Launch Performance

> Fictional eval note for fitcheck.

> Summary
> "Cold launch on older iPhones took over three seconds. I profiled it with Instruments, moved work off the main thread and deferred non-critical setup, and cold launch dropped from 3.1 seconds to 1.2 seconds."

## What I did

- Recorded App Launch and Time Profiler traces in Instruments on the three oldest supported devices.
- Found that analytics setup, font loading and a large JSON parse all ran on the main thread before the first frame.
- Deferred non-critical SDK setup, lazy-loaded fonts and moved parsing to a background actor.
- Added a launch-time check to Xcode Cloud so a regression over 10% fails the build.

## Result

- Cold launch fell from 3.1 seconds to 1.2 seconds.
- Scroll hitches on the trail list dropped by half after the same profiling pass.
