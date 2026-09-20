# Large-area generation: development notes

Status: **done**. Written 2026-09-20, AI-assisted (Claude), reviewed and directed by monlum58 throughout — every claim below is backed by a real command run and its actual output, not inferred.

## The problem

Minecraft world generation from real-world OSM + elevation data hits a hard RAM wall at metro scale. Two data points made this concrete before any code was touched:

- A separate hobby project of monlum's, `tokyo-minecraft`, had already hit this exactly: an in-memory voxel `HashMap` holding every block until final export worked fine for one 600m tile but was never pushed further, and its own design notes left the real fix undecided.
- Upstream `louis-e/arnis` has a filed issue reporting 58GB+ RAM and a crash the moment the requested area gets big (their issue #723).

Arnis itself is the better base to build on — general-purpose, downloads its own OSM + elevation data for any bounding box — but had never been pushed past normal city-district scale either.

## Goal

Generate a whole metro-scale bounding box (order 1,000+ km², "a whole Tokyo") to completion on ordinary desktop RAM, with the generated world visually and mechanically identical to what unmodified Arnis would produce for any area that already worked before this change. No change to block palette, terrain algorithms, or rendering logic — memory behavior only.

Explicitly out of scope: distributed/multi-machine generation, upstreaming to `louis-e/arnis` (this is a personal fork), and Overpass rate-limit policy design beyond reusing Arnis's own existing multi-server fallback.

## What was found and fixed

**1. OSM query tiling.** Past a size threshold, Arnis's single Overpass query got truncated by every mirror and hard-errored. Fixed by splitting an oversized bounding box into a grid of sub-queries (`LLBBox::split_into_grid`), each fetched independently through Arnis's existing multi-server fallback, then merged by `(type, id)` with duplicates dropped. Verified: a real ~430km² Tokyo-area bbox that used to hard-error now completes; a synthetic-grid unit test proves the tiling has no gaps; a merge unit test proves no duplicate or dropped elements at tile seams.

A parity test comparing a tiled fetch against a single untiled fetch initially looked broken (951 mismatched elements), but the real cause was two things, neither a tiling bug: (a) Overpass legitimately re-emits an element once per relation it belongs to, and the code already documented handling that downstream — the test just wasn't deduping the same way before comparing; (b) one way and a few of its nodes were present in the single-query result but missing from the tiled fetch because two independent public Overpass mirrors answered the two requests and hadn't synced from the OSM edit stream at exactly the same moment (confirmed by querying each mirror directly and comparing their `timestamp_osm_base`). Fixed the test to dedupe correctly and accept a small tolerance for genuine cross-mirror staleness, with the reasoning written into the test itself.

**2. Elevation pipeline memory.** This was the real remaining scaling risk. Every elevation buffer in the pipeline — the raw fetch grid, the terrain anomaly-repair snapshot, the land-cover Gaussian blur buffers, the final scale-to-Minecraft output — was a plain in-memory array, sized to the whole requested area, with no ceiling. Real `/usr/bin/time -v` measurements showed RSS growing faster than area (967MB at 25km², 5.13GB at 225km²), confirming this would land in the tens-of-GB range for a true metro-scale request.

Fixed by writing a small `MmapGrid<T>` type (`src/elevation/mmap_grid.rs`) — a 2D grid backed by a memory-mapped temp file instead of a `Vec<Vec<T>>`, generic over `f32`/`f64`, with the exact same `[row][col]` indexing so almost no call site needed rewriting, plus parallel-iteration support built on the existing `rayon` dependency. Every full-size buffer in the elevation pipeline (all five data providers, every post-processing pass, the final output) now uses it.

Two real bugs surfaced while getting this right, not assumed away:

- **The mmap needs to be backed by real disk, not the OS's default temp directory.** `tempfile::tempfile()` defaults to `/tmp`, and on the machine this was built on (and plenty of others), `/tmp` is itself a RAM-backed filesystem (`tmpfs`). A "disk-backed" mmap over a RAM filesystem is backed by RAM all along — there's nothing for the kernel to evict it to, which quietly defeats the entire point. Fixed by pointing the backing files at the OS cache directory instead (confirmed with `df -T` to actually be a real disk).
- **The vertical half of every Gaussian blur pass had a disk-hostile access pattern.** It built each output column by reading one value from every row of the grid — for a disk-backed, row-major grid, that's close to the worst possible pattern, rereading gigabytes per call. This is what made an early constrained-memory test run for 30+ minutes with no end in sight. Fixed with a blocked/tiled matrix transpose (`MmapGrid::transpose()`, 512×512 tiles, bounded working set regardless of grid size) — the standard technique for filtering a large row-major raster in the other axis: transpose, run the same row-wise blur the horizontal pass already used, transpose back.

## Verification

- Every change was checked block-for-block against a pre-change build using a purpose-built comparator (now `examples/world_block_diff.rs` in this repo) that reads two generated worlds and diffs every block. Result across every slice of this work: **0 differing block positions out of 402,653,184 compared**, each time.
- The full non-network test suite stayed green throughout (1191 tests passing at the end).
- Real memory-constrained runs, not just theory: a `systemd-run --scope -p MemoryMax=6G` cgroup cap on a real 225km² bounding box completed end-to-end in 3 minutes 3 seconds, peak RSS 5.99GB — versus the unmodified baseline, which needed 5.96GB *unconstrained* (no cap at all) for the same area, and would instantly fail under any real cap that tight.
- **Real-world confirmation at true target scale**: monlum ran a ~2,000km² bounding box (central Tokyo through most of the 23 wards, full detail — real buildings and roads, not just terrain) on his own 32GB machine. It completed in 40 minutes (22 of which were downloading OSM/elevation/Overture data), peak memory around 20GB RAM plus swap — on a rerun with closer monitoring, swap filled too, leaving only about 1.5GB of combined RAM+swap headroom at the tightest point. Output world folder: ~33GB. No crash, loaded and played correctly in Minecraft. That's an area upstream Arnis cannot attempt at all — the entire reason this fork exists.
- **Honest limit of what this work bounds.** The mmap fix targets the elevation/post-processing pipeline specifically, which is genuinely bounded now (see the 6GB-cap proof above). It does not bound everything: the parsed OSM element list, the Overture building data, and the land-cover/canopy grids are all still plain in-memory arrays that scale with area and were out of scope for this pass (see "Remaining ideas" below). At ~2,000km² those are large enough that the swap usage above is plausibly coming from them, not from the elevation pipeline itself. Practically: this specific area is already close to this specific machine's ceiling, and there's no guarantee a meaningfully larger one succeeds without either more memory or extending the same mmap treatment to those other structures.

## Also shipped along the way

- **`--name <NAME>` CLI flag** — the GUI already had a way to name a generated world; the CLI didn't. Wired the CLI path to the existing (already-tested) naming logic instead of writing anything new.
- **`examples/world_block_diff.rs`** — the block-diff verification tool used throughout this work, formalized into the repo so a future change can be checked the same way without rebuilding the tool from scratch.
- **Fork labeling** — README, `NOTICE` (per Apache License 2.0 §4(b), which requires modified files to carry notice that they changed), and the CLI's own banner/`--help` output all now make clear this is monlum58's fork, with credit to original author Louis Erbkamm kept prominent throughout.

## Remaining ideas (not blocking, not done)

- `LandCoverData`'s three grids and `CanopyData` are still plain in-memory arrays — smaller per-cell than what was just fixed (1-4 bytes vs. 4-8), and the real metro-scale run above already proves the current fix is sufficient at the actual target scale. Not worth converting speculatively; two earlier guesses about which buffer would matter most were both wrong before the real fix (the blur access pattern) was found by actually measuring.
- Whether to ever propose any of this back to upstream `louis-e/arnis` is an open, deliberately deferred question — real evidence now exists that it works, but that's a separate decision from finishing this fork's own goal.
