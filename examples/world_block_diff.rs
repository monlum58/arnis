//! Block-for-block diff between two Java Edition worlds, used to verify a
//! code change didn't alter generated terrain/objects.
//!
//! Grew out of the ISC-11 large-area-generation work (mmap-backing the
//! elevation pipeline): every storage-layer change in that effort was
//! checked against a pre-change build with this same technique before being
//! trusted, since "no change to the look of generated worlds" was a hard
//! constraint that only a real block-by-block comparison can actually prove
//! — a passing test suite proves the code paths run, not that two builds
//! produce the same world.
//!
//! Usage:
//!   cargo run --release --example world_block_diff -- <old_world_dir> <new_world_dir>
//!
//! Both paths are Java Edition world directories (the folder containing
//! `level.dat` and `region/`), typically two runs of `arnis` over the same
//! `--bbox` from different builds. Exits non-zero if any block position
//! differs, so this is usable as a CI-style gate as well as an interactive
//! check.

use fastanvil::{Chunk, JavaChunk, Region};
use std::collections::HashSet;
use std::fs::File;
use std::path::{Path, PathBuf};

fn region_files(world_dir: &Path) -> Vec<(isize, isize, PathBuf)> {
    let mut out = vec![];
    let region_dir = world_dir.join("region");
    for entry in std::fs::read_dir(&region_dir)
        .unwrap_or_else(|e| panic!("read region dir {}: {e}", region_dir.display()))
    {
        let entry = entry.expect("read region dir entry");
        let name = entry.file_name().into_string().expect("region filename is valid UTF-8");
        // r.<x>.<z>.mca
        let parts: Vec<&str> = name.trim_end_matches(".mca").split('.').collect();
        if parts.len() == 3 && parts[0] == "r" {
            let x: isize = parts[1].parse().expect("region x coordinate");
            let z: isize = parts[2].parse().expect("region z coordinate");
            out.push((x, z, entry.path()));
        }
    }
    out.sort();
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: world_block_diff <old_world_dir> <new_world_dir>");
        std::process::exit(2);
    }
    let old_dir = Path::new(&args[1]);
    let new_dir = Path::new(&args[2]);

    let old_regions = region_files(old_dir);
    let new_regions = region_files(new_dir);

    let old_keys: HashSet<(isize, isize)> = old_regions.iter().map(|(x, z, _)| (*x, *z)).collect();
    let new_keys: HashSet<(isize, isize)> = new_regions.iter().map(|(x, z, _)| (*x, *z)).collect();
    let mut mismatch = old_keys != new_keys;
    if mismatch {
        println!(
            "REGION SET MISMATCH: old-only={:?} new-only={:?}",
            old_keys.difference(&new_keys).collect::<Vec<_>>(),
            new_keys.difference(&old_keys).collect::<Vec<_>>()
        );
    }

    let mut total_compared: u64 = 0;
    let mut total_diffs: u64 = 0;
    let mut diff_examples: Vec<String> = vec![];
    let mut old_only_nonair: u64 = 0;
    let mut new_only_nonair: u64 = 0;

    for (rx, rz, old_path) in &old_regions {
        let new_path = new_dir.join("region").join(format!("r.{rx}.{rz}.mca"));
        if !new_path.exists() {
            continue;
        }
        let mut old_region =
            Region::from_stream(File::open(old_path).expect("open old region file")).unwrap();
        let mut new_region =
            Region::from_stream(File::open(&new_path).expect("open new region file")).unwrap();

        for cx in 0..32usize {
            for cz in 0..32usize {
                let old_bytes = old_region.read_chunk(cx, cz).expect("read old chunk");
                let new_bytes = new_region.read_chunk(cx, cz).expect("read new chunk");
                let (old_bytes, new_bytes) = match (old_bytes, new_bytes) {
                    (Some(o), Some(n)) => (o, n),
                    (None, None) => continue,
                    _ => {
                        diff_examples
                            .push(format!("chunk r.{rx}.{rz} ({cx},{cz}): one side missing entirely"));
                        total_diffs += 1;
                        continue;
                    }
                };
                let old_chunk = JavaChunk::from_bytes(&old_bytes).expect("parse old chunk");
                let new_chunk = JavaChunk::from_bytes(&new_bytes).expect("parse new chunk");

                let old_y = old_chunk.y_range();
                let new_y = new_chunk.y_range();
                let y_start = old_y.start.max(new_y.start);
                let y_end = old_y.end.min(new_y.end);

                for lx in 0..16usize {
                    for lz in 0..16usize {
                        for y in y_start..y_end {
                            let on = old_chunk.block(lx, y, lz).map(|b| b.name()).unwrap_or("air");
                            let nn = new_chunk.block(lx, y, lz).map(|b| b.name()).unwrap_or("air");
                            total_compared += 1;
                            if on != nn {
                                total_diffs += 1;
                                if diff_examples.len() < 20 {
                                    let wx = *rx * 512 + (cx as isize) * 16 + lx as isize;
                                    let wz = *rz * 512 + (cz as isize) * 16 + lz as isize;
                                    diff_examples.push(format!("({wx},{y},{wz}): old={on} new={nn}"));
                                }
                            }
                        }
                        // Blocks outside the overlapping Y range never enter the loop above,
                        // so a non-air block stranded there would otherwise go unnoticed.
                        for y in old_y.clone() {
                            if (y < y_start || y >= y_end)
                                && old_chunk
                                    .block(lx, y, lz)
                                    .is_some_and(|b| !matches!(b.name(), "air" | "cave_air"))
                            {
                                old_only_nonair += 1;
                            }
                        }
                        for y in new_y.clone() {
                            if (y < y_start || y >= y_end)
                                && new_chunk
                                    .block(lx, y, lz)
                                    .is_some_and(|b| !matches!(b.name(), "air" | "cave_air"))
                            {
                                new_only_nonair += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    println!("total block positions compared: {total_compared}");
    println!("total differing positions: {total_diffs}");
    println!("non-air blocks in old-only Y range (outside overlap): {old_only_nonair}");
    println!("non-air blocks in new-only Y range (outside overlap): {new_only_nonair}");
    for d in &diff_examples {
        println!("  {d}");
    }

    mismatch |= total_diffs > 0 || old_only_nonair > 0 || new_only_nonair > 0;
    if mismatch {
        std::process::exit(1);
    }
}
