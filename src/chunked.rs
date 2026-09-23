//! Chunked generation: one large area built as a grid of smaller runs, each its
//! own `arnis` process so every chunk's memory is released before the next
//! starts, then stitched into a single Java world.
//!
//! A chunk run covers its owned regions plus a margin of whole regions on every
//! side. It shares the full area's block spacing (`--reference-bbox`), height
//! range (`--elevation-range`) and a disjoint block of signage map ids
//! (`--map-id-base`), so every block lands where one run over the full area
//! would put it. Only the regions it owns are kept; the
//! margin exists so seam content (terrain smoothing, buildings crossing the
//! boundary, trees overhanging it) is computed with its neighbourhood present.
//!
//! Chunk boundaries sit on 512-block region boundaries, so no two chunks ever
//! write the same region file and merging is copying the files each one owns.

use crate::args::Args;
use crate::coordinate_system::transformation::CoordTransformer;
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const REGION_BLOCKS: i32 = 512;

/// Flags the driver sets per chunk; anything else on the command line passes
/// through to every chunk unchanged. `true` = the flag takes a value.
const DRIVER_FLAGS: &[(&str, bool)] = &[
    ("--bbox", true),
    ("--output-dir", true),
    ("--path", true),
    ("--name", true),
    ("--reference-bbox", true),
    ("--elevation-range", true),
    ("--probe-elevation", false),
    ("--terrain-base", true),
    ("--map-id-base", true),
    ("--chunk-regions", true),
    ("--chunk-margin-regions", true),
];

/// One chunk: the regions it owns and the block range it generates.
#[derive(Debug, Clone, PartialEq)]
struct ChunkPlan {
    index: usize,
    /// Owned regions, half-open, in the full world's region coordinates.
    region_x: (i32, i32),
    region_z: (i32, i32),
    /// Generated block range (owned + margin), half-open at the end, in the
    /// full world's block coordinates. `block_x.0`/`block_z.0` are region
    /// multiples: they are the chunk's origin in the world.
    block_x: (i32, i32),
    block_z: (i32, i32),
}

/// Splits a world spanning blocks `0..=max_x` by `0..=max_z` into chunks of
/// `chunk_regions` x `chunk_regions` owned regions, each generated with
/// `margin_regions` extra regions on every side where the world allows.
fn plan_chunks(max_x: i32, max_z: i32, chunk_regions: i32, margin_regions: i32) -> Vec<ChunkPlan> {
    let regions_x = max_x / REGION_BLOCKS + 1;
    let regions_z = max_z / REGION_BLOCKS + 1;
    let mut plans = Vec::new();
    let mut rz = 0;
    while rz < regions_z {
        let rz_end = (rz + chunk_regions).min(regions_z);
        let mut rx = 0;
        while rx < regions_x {
            let rx_end = (rx + chunk_regions).min(regions_x);
            let bx0 = (rx - margin_regions).max(0) * REGION_BLOCKS;
            let bz0 = (rz - margin_regions).max(0) * REGION_BLOCKS;
            let bx1 = ((rx_end + margin_regions) * REGION_BLOCKS).min(max_x);
            let bz1 = ((rz_end + margin_regions) * REGION_BLOCKS).min(max_z);
            plans.push(ChunkPlan {
                index: plans.len(),
                region_x: (rx, rx_end),
                region_z: (rz, rz_end),
                block_x: (bx0, bx1),
                block_z: (bz0, bz1),
            });
            rx = rx_end;
        }
        rz = rz_end;
    }
    plans
}

fn parse_region_name(name: &str) -> Option<(i32, i32)> {
    let mut parts = name.strip_prefix("r.")?.strip_suffix(".mca")?.split('.');
    let x = parts.next()?.parse().ok()?;
    let z = parts.next()?.parse().ok()?;
    parts.next().is_none().then_some((x, z))
}

/// Map id in a `map_<id>.dat` file name.
fn parse_map_id(name: &str) -> Option<i32> {
    name.strip_prefix("map_")?
        .strip_suffix(".dat")?
        .parse()
        .ok()
}

/// The command line minus the flags the driver sets per chunk.
fn passthrough_args() -> Vec<String> {
    let mut out = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let flag = arg.split('=').next().unwrap_or(&arg);
        match DRIVER_FLAGS.iter().find(|(f, _)| *f == flag) {
            Some((_, takes_value)) => {
                if *takes_value && !arg.contains('=') {
                    args.next();
                }
            }
            None => out.push(arg),
        }
    }
    out
}

fn bbox_arg(t: &CoordTransformer, plan: &ChunkPlan) -> String {
    let (max_lat, min_lng) = t.reference_to_latlng(plan.block_x.0 as f64, plan.block_z.0 as f64);
    let (min_lat, max_lng) = t.reference_to_latlng(plan.block_x.1 as f64, plan.block_z.1 as f64);
    format!("{min_lat},{min_lng},{max_lat},{max_lng}")
}

fn fail(msg: impl std::fmt::Display) -> ! {
    eprintln!("{} {msg}", "Error:".red().bold());
    std::process::exit(1);
}

/// What a chunked run was started with. A resume must match it exactly, or the
/// chunks already merged would not fit the ones still to come.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct RunKey {
    bbox: String,
    scale: f64,
    chunk_regions: i32,
    margin_regions: i32,
}

/// Progress of a chunked run, saved after every step so the same command can
/// pick up where a crash, reboot or unplugged drive left it. Chunks run in
/// order, so a count of finished ones is enough.
#[derive(Debug, Serialize, Deserialize)]
struct RunState {
    key: RunKey,
    final_name: String,
    probed: bool,
    elevation_range: Option<(f64, f64)>,
    terrain_base: Option<i32>,
    next_map_id: i32,
    chunks_done: usize,
}

impl RunState {
    fn load(path: &Path) -> Option<Self> {
        let text = fs::read_to_string(path).ok()?;
        Some(serde_json::from_str(&text).unwrap_or_else(|e| {
            fail(format!("unreadable run state {}: {e}", path.display()))
        }))
    }

    /// Written to a temp file and renamed, so a crash mid-write cannot leave a
    /// half state behind.
    fn save(&self, path: &Path) {
        let tmp = path.with_extension("json.tmp");
        let text = serde_json::to_string_pretty(self).expect("serialize run state");
        fs::write(&tmp, text)
            .and_then(|_| fs::rename(&tmp, path))
            .unwrap_or_else(|e| fail(format!("save run state {}: {e}", path.display())));
    }
}

/// A chunk (or its probe) gets this many tries: a child that exhausted every
/// Overpass mirror once often succeeds a minute later, and one flaky request
/// should not stop a run that takes a day.
const CHUNK_ATTEMPTS: u32 = 3;

fn with_retries<T>(label: &str, mut attempt: impl FnMut() -> Result<T, String>) -> T {
    for n in 1..=CHUNK_ATTEMPTS {
        match attempt() {
            Ok(v) => return v,
            Err(e) if n < CHUNK_ATTEMPTS => {
                eprintln!(
                    "{} {label} failed ({e}); retrying in 60 s ({n}/{CHUNK_ATTEMPTS})",
                    "Warning:".yellow().bold()
                );
                std::thread::sleep(std::time::Duration::from_secs(60));
            }
            Err(e) => fail(format!(
                "{label} failed {CHUNK_ATTEMPTS} times ({e}). Run the same command again to resume."
            )),
        }
    }
    unreachable!()
}

/// Runs the whole chunked generation and exits.
pub fn run(args: &Args) -> ! {
    let bbox = args
        .bbox
        .unwrap_or_else(|| fail("--chunk-regions needs --bbox."));
    if args.bedrock || args.luanti {
        fail("--chunk-regions only supports Java worlds.");
    }
    if args.rotation != 0.0 {
        fail("--chunk-regions cannot be combined with --rotation.");
    }
    if args.reference_bbox.is_some() {
        fail("--chunk-regions sets --reference-bbox itself; drop it.");
    }
    if args.spawn_lat.is_some() || args.spawn_lng.is_some() {
        fail("--chunk-regions does not support --spawn-lat/--spawn-lng yet.");
    }
    if args.save_json_file.is_some() {
        fail("--chunk-regions cannot write --save-json-file; each chunk fetches its own area.");
    }
    let output_dir = args
        .path
        .clone()
        .unwrap_or_else(|| fail("--chunk-regions needs --output-dir."));
    let chunk_regions = args.chunk_regions.unwrap_or(1).max(1) as i32;
    let margin_regions = args.chunk_margin_regions as i32;

    let (reference, whole) = CoordTransformer::llbbox_to_xzbbox(&bbox, args.scale)
        .unwrap_or_else(|e| fail(format!("Invalid --bbox: {e}")));
    let plans = plan_chunks(whole.max_x(), whole.max_z(), chunk_regions, margin_regions);
    let exe = std::env::current_exe().unwrap_or_else(|e| fail(format!("current_exe: {e}")));
    let passthrough = passthrough_args();

    fs::create_dir_all(&output_dir).unwrap_or_else(|e| fail(format!("create output dir: {e}")));
    let requested_name = args
        .name
        .clone()
        .unwrap_or_else(|| "Arnis World".to_string());

    let ref_arg = format!(
        "{},{},{},{}",
        bbox.min().lat(),
        bbox.min().lng(),
        bbox.max().lat(),
        bbox.max().lng()
    );
    let key = RunKey {
        bbox: ref_arg.clone(),
        scale: args.scale,
        chunk_regions,
        margin_regions,
    };

    // Keyed on the name asked for, not the de-duplicated one: rerunning the same
    // command must find the unfinished run rather than start a "(2)" beside it.
    let safe_name: String = requested_name
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' { c } else { '_' })
        .collect();
    let scratch = output_dir.join(format!(".{safe_name}.chunks"));
    fs::create_dir_all(&scratch).unwrap_or_else(|e| fail(format!("create scratch dir: {e}")));
    let state_path = scratch.join("state.json");

    let mut state = match RunState::load(&state_path) {
        Some(state) if state.key == key => {
            println!(
                "{} {}/{} chunk(s) already done; resuming.",
                "Resuming:".bold(),
                state.chunks_done,
                plans.len()
            );
            state
        }
        Some(_) => fail(format!(
            "{} holds an unfinished chunked run with a different area or settings. \
             Finish it with its original command, delete that folder, or use another --name.",
            scratch.display()
        )),
        None => {
            let state = RunState {
                key,
                final_name: crate::world_utils::generate_unique_custom_world_name(
                    &output_dir,
                    &requested_name,
                ),
                probed: false,
                elevation_range: args.elevation_range,
                terrain_base: args.terrain_base,
                next_map_id: crate::decals::registry::DecalRegistry::FIRST_ID,
                chunks_done: 0,
            };
            state.save(&state_path);
            state
        }
    };
    let final_world = output_dir.join(&state.final_name);

    println!(
        "{} {} x {} blocks as {} chunk(s) of up to {chunk_regions}x{chunk_regions} regions \
         (margin {margin_regions} region(s)).",
        "Chunked generation:".bold(),
        whole.max_x() + 1,
        whole.max_z() + 1,
        plans.len()
    );

    // Phase 1: one height range and one terrain base for the whole world. The
    // base is the highest any chunk needs: each derives it from the deepest
    // water it must carve, and a higher base only leaves more room below.
    let needs_probe =
        args.terrain() && (state.elevation_range.is_none() || state.terrain_base.is_none());
    if needs_probe && !state.probed {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        let mut base = i32::MIN;
        for plan in &plans {
            println!(
                "{} measuring terrain height, chunk {}/{}...",
                "[chunks]".bold(),
                plan.index + 1,
                plans.len()
            );
            let label = format!("terrain probe for chunk {}", plan.index + 1);
            let (range, chunk_base) = with_retries(&label, || {
                let output = Command::new(&exe)
                    .args(&passthrough)
                    .arg(format!("--bbox={}", bbox_arg(&reference, plan)))
                    .arg(format!("--reference-bbox={ref_arg}"))
                    .arg("--probe-elevation")
                    .arg("--output-dir")
                    .arg(&scratch)
                    .stderr(Stdio::inherit())
                    .output()
                    .map_err(|e| e.to_string())?;
                if !output.status.success() {
                    return Err(output.status.to_string());
                }
                parse_probe(&String::from_utf8_lossy(&output.stdout))
            });
            lo = lo.min(range.0);
            hi = hi.max(range.1);
            base = base.max(chunk_base);
        }
        state.elevation_range.get_or_insert((lo, hi));
        state.terrain_base.get_or_insert(base);
        state.probed = true;
        state.save(&state_path);
    }
    if let (Some((lo, hi)), Some(base)) = (state.elevation_range, state.terrain_base) {
        println!(
            "{} shared height range {lo:.1} m .. {hi:.1} m, terrain base y={base}",
            "[chunks]".bold()
        );
    }

    // Phase 2: generate each chunk and move its owned regions into the world.
    for plan in plans.iter().skip(state.chunks_done) {
        println!(
            "{} generating chunk {}/{} (regions x {}..{}, z {}..{})",
            "[chunks]".bold(),
            plan.index + 1,
            plans.len(),
            plan.region_x.0,
            plan.region_x.1 - 1,
            plan.region_z.0,
            plan.region_z.1 - 1
        );
        let chunk_out = scratch.join(format!("chunk-{}", plan.index));
        let label = format!("chunk {}", plan.index + 1);
        with_retries(&label, || {
            if chunk_out.exists() {
                fs::remove_dir_all(&chunk_out).map_err(|e| format!("clear chunk dir: {e}"))?;
            }
            fs::create_dir_all(&chunk_out).map_err(|e| format!("create chunk dir: {e}"))?;
            let mut cmd = Command::new(&exe);
            cmd.args(&passthrough)
                .arg(format!("--bbox={}", bbox_arg(&reference, plan)))
                .arg(format!("--reference-bbox={ref_arg}"))
                .arg(format!("--map-id-base={}", state.next_map_id))
                .arg("--output-dir")
                .arg(&chunk_out)
                .arg("--name")
                .arg("chunk");
            if let Some((lo, hi)) = state.elevation_range {
                cmd.arg(format!("--elevation-range={lo},{hi}"));
            }
            if let Some(base) = state.terrain_base {
                cmd.arg(format!("--terrain-base={base}"));
            }
            let status = cmd.status().map_err(|e| e.to_string())?;
            if status.success() {
                Ok(())
            } else {
                Err(status.to_string())
            }
        });

        // A crash between the first chunk's move and its state save leaves a
        // partial base world behind; nothing else can be in it yet.
        if plan.index == 0 && final_world.exists() {
            fs::remove_dir_all(&final_world)
                .unwrap_or_else(|e| fail(format!("clear partial world: {e}")));
        }
        let chunk_world = chunk_out.join("chunk");
        state.next_map_id = merge_chunk(&chunk_world, &final_world, plan, state.next_map_id)
            .unwrap_or_else(|e| fail(format!("merge chunk {}: {e}", plan.index + 1)));
        state.chunks_done = plan.index + 1;
        state.save(&state_path);
        let _ = fs::remove_dir_all(&chunk_out);
    }

    if let Err(e) = crate::world_utils::update_level_name(&final_world, &state.final_name) {
        eprintln!("Warning: could not set the world name: {e}");
    }
    let _ = fs::remove_dir_all(&scratch);
    println!(
        "{} {} chunk(s) merged into {}",
        "Done:".green().bold(),
        plans.len(),
        final_world.display()
    );
    std::process::exit(0);
}

/// The height range and terrain base a `--probe-elevation` child printed.
fn parse_probe(stdout: &str) -> Result<((f64, f64), i32), String> {
    let range = stdout
        .lines()
        .find_map(|l| l.strip_prefix("ARNIS_ELEVATION_RANGE "))
        .ok_or("probe printed no range")?;
    let mut it = range.split_whitespace().map(str::parse::<f64>);
    let (Some(Ok(lo)), Some(Ok(hi))) = (it.next(), it.next()) else {
        return Err(format!("bad probe range: {range}"));
    };
    let base = stdout
        .lines()
        .find_map(|l| l.strip_prefix("ARNIS_TERRAIN_BASE "))
        .and_then(|v| v.trim().parse::<i32>().ok())
        .ok_or("probe printed no base")?;
    Ok(((lo, hi), base))
}

/// Moves one finished chunk's owned regions (and its signage maps) into the
/// world. Region files already carry world coordinates, so this is a copy. The
/// first chunk becomes the base: level.dat, spawn, datapacks and the preview
/// map come from it.
/// Returns the next free map id.
fn merge_chunk(
    chunk_world: &Path,
    final_world: &Path,
    plan: &ChunkPlan,
    map_id_base: i32,
) -> Result<i32, String> {
    let owns = |rx: i32, rz: i32| {
        (plan.region_x.0..plan.region_x.1).contains(&rx)
            && (plan.region_z.0..plan.region_z.1).contains(&rz)
    };

    let chunk_regions = chunk_world.join("region");
    let region_files: Vec<PathBuf> = fs::read_dir(&chunk_regions)
        .map_err(|e| format!("read {}: {e}", chunk_regions.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();

    if plan.index == 0 {
        fs::rename(chunk_world, final_world).map_err(|e| format!("move base world: {e}"))?;
        let region_dir = final_world.join("region");
        for path in fs::read_dir(&region_dir).map_err(|e| e.to_string())? {
            let path = path.map_err(|e| e.to_string())?.path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            match parse_region_name(name) {
                Some((rx, rz)) if owns(rx, rz) => {}
                _ => fs::remove_file(&path).map_err(|e| e.to_string())?,
            }
        }
        return next_free_map_id(&final_world.join("data"), map_id_base);
    }

    let region_dir = final_world.join("region");
    for path in region_files {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if let Some((rx, rz)) = parse_region_name(name) {
            if owns(rx, rz) {
                fs::copy(&path, region_dir.join(name)).map_err(|e| e.to_string())?;
            }
        }
    }

    // Signage maps got ids from map_id_base up, so they cannot collide. The
    // preview and branding maps below it describe this chunk alone.
    let data_src = chunk_world.join("data");
    let data_dst = final_world.join("data");
    if data_src.is_dir() {
        fs::create_dir_all(&data_dst).map_err(|e| e.to_string())?;
        for entry in fs::read_dir(&data_src).map_err(|e| e.to_string())? {
            let path = entry.map_err(|e| e.to_string())?.path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if parse_map_id(name).is_some_and(|id| id >= map_id_base) {
                fs::copy(&path, data_dst.join(name)).map_err(|e| e.to_string())?;
            }
        }
    }
    next_free_map_id(&data_dst, map_id_base)
}

fn next_free_map_id(data_dir: &Path, at_least: i32) -> Result<i32, String> {
    let mut next = at_least;
    if let Ok(entries) = fs::read_dir(data_dir) {
        for entry in entries.flatten() {
            if let Some(id) = entry.file_name().to_str().and_then(parse_map_id) {
                next = next.max(id + 1);
            }
        }
    }
    Ok(next)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn plan_owns_every_region_exactly_once_and_origins_are_region_aligned() {
        let (max_x, max_z) = (5 * 512 + 100, 3 * 512 + 7);
        let plans = plan_chunks(max_x, max_z, 2, 1);
        let mut owner: HashMap<(i32, i32), usize> = HashMap::new();
        for p in &plans {
            assert_eq!(p.block_x.0 % 512, 0);
            assert_eq!(p.block_z.0 % 512, 0);
            for rx in p.region_x.0..p.region_x.1 {
                for rz in p.region_z.0..p.region_z.1 {
                    assert!(
                        owner.insert((rx, rz), p.index).is_none(),
                        "region owned twice"
                    );
                    // Owned regions lie inside the generated range, margin included.
                    assert!(rx * 512 >= p.block_x.0 && rz * 512 >= p.block_z.0);
                }
            }
        }
        assert_eq!(owner.len(), 6 * 4, "every region owned");
        assert_eq!((plans[0].block_x.0, plans[0].block_z.0), (0, 0));
    }

    #[test]
    fn passthrough_region_and_map_names_parse() {
        assert_eq!(parse_region_name("r.-2.13.mca"), Some((-2, 13)));
        assert_eq!(parse_region_name("r.1.mca"), None);
        assert_eq!(parse_map_id("map_42.dat"), Some(42));
        assert_eq!(parse_map_id("idcounts.dat"), None);
    }

    #[test]
    fn probe_output_parses_and_rejects_missing_lines() {
        let out = "noise\nARNIS_ELEVATION_RANGE -3.5 2100.25\nARNIS_TERRAIN_BASE -12\n";
        assert_eq!(parse_probe(out), Ok(((-3.5, 2100.25), -12)));
        assert!(parse_probe("ARNIS_ELEVATION_RANGE 1 2\n").is_err());
        assert!(parse_probe("ARNIS_TERRAIN_BASE 4\n").is_err());
    }

    #[test]
    fn run_state_survives_a_save_and_load() {
        let dir = std::env::temp_dir().join(format!("arnis-state-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.json");
        let state = RunState {
            key: RunKey {
                bbox: "1,2,3,4".into(),
                scale: 1.0,
                chunk_regions: 16,
                margin_regions: 1,
            },
            final_name: "World (2)".into(),
            probed: true,
            elevation_range: Some((-1.0, 3000.0)),
            terrain_base: Some(-20),
            next_map_id: 57,
            chunks_done: 3,
        };
        state.save(&path);
        let back = RunState::load(&path).unwrap();
        assert_eq!(back.key, state.key);
        assert_eq!(back.final_name, "World (2)");
        assert_eq!(back.terrain_base, Some(-20));
        assert_eq!(back.next_map_id, 57);
        assert_eq!(back.chunks_done, 3);
        assert!(!path.with_extension("json.tmp").exists());
        fs::remove_file(&path).unwrap();
        fs::remove_dir(&dir).unwrap();
    }
}
