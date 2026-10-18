//! Input encoding for MPC programs.

use std::cell::Cell;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use super::{extract_reuse, CaseData, SpdzConfig};

/// The live run's teardown state, so the signal handler runs what `Drop` runs.
static ACTIVE_CLEANUP: Mutex<Option<(PathBuf, bool)>> = Mutex::new(None);

/// Remove `Player-Data/Memory-<type>-P<i>`: MP-SPDZ dumps the tape's global
/// `sint` memory there at the end of every run unless `no_memory_output` is
/// passed — the encoding matrices and Y as raw shares. A binary that silently
/// ignores an unknown `-o` value still writes it, so the glob is the net.
fn remove_memory_dumps(player_data: &Path, cleaned: &mut Vec<String>) {
    if let Ok(entries) = fs::read_dir(player_data) {
        for e in entries.filter_map(|e| e.ok()) {
            let name = e.file_name().to_string_lossy().into_owned();
            if name.starts_with("Memory-")
                && name.contains("-P")
                && fs::remove_file(e.path()).is_ok()
            {
                cleaned.push(name);
            }
        }
    }
}

/// The teardown itself, shared by the `Drop` guard and the signal handler.
/// Returns the names it acted on (for the caller's log line).
pub(crate) fn teardown_secret_material(
    spdz_path: &Path,
    preserve_persistence: bool,
) -> Vec<String> {
    let mut cleaned: Vec<String> = Vec::new();
    let player_data = spdz_path.join("Player-Data");
    // Raw event streams: always truncated (never needed by a tail run).
    if let Ok(entries) = fs::read_dir(&player_data) {
        for e in entries.filter_map(|e| e.ok()) {
            let name = e.file_name().to_string_lossy().into_owned();
            if name.starts_with("Input-P") && name.ends_with("-0") && File::create(e.path()).is_ok()
            {
                cleaned.push(name);
            }
        }
    }
    // End-of-run share dump: removed on every path, reuse included.
    remove_memory_dumps(&player_data, &mut cleaned);
    if preserve_persistence {
        return cleaned;
    }
    // Secret-share slab + sidecar: removed on non-reuse or any error path.
    if let Ok(entries) = fs::read_dir(spdz_path.join("Persistence")) {
        for e in entries.filter_map(|e| e.ok()) {
            let name = e.file_name().to_string_lossy().into_owned();
            if name.starts_with("Transactions-P")
                && name.ends_with(".data")
                && fs::remove_file(e.path()).is_ok()
            {
                cleaned.push(name);
            }
        }
    }
    if fs::remove_file(extract_reuse::sidecar_path(spdz_path)).is_ok() {
        cleaned.push("extract_slab.json".to_string());
    }
    cleaned
}

/// Install the SIGINT/SIGTERM/SIGHUP teardown: kill the party children, run the
/// same wipe `SecretsCleanup::drop` runs, exit 130. `ctrlc` runs the closure on
/// its own thread, so this file I/O is not in signal context.
pub(crate) fn install_signal_teardown() {
    let installed = ctrlc::set_handler(|| {
        eprintln!(
            "\n[SPDZ] Signal received: killing MP-SPDZ parties and wiping per-run \
             secret material"
        );
        super::runtime::kill_party_children();
        let state = ACTIVE_CLEANUP.lock().unwrap_or_else(|e| e.into_inner()).clone();
        if let Some((path, preserve)) = state {
            let cleaned = teardown_secret_material(&path, preserve);
            eprintln!(
                "[SPDZ] Signal teardown cleaned: {}{}",
                if cleaned.is_empty() { "nothing".into() } else { cleaned.join(", ") },
                if preserve { " (extract slab PRESERVED)" } else { "" }
            );
        }
        std::process::exit(130);
    });
    if let Err(e) = installed {
        eprintln!("[SPDZ] WARNING: no signal teardown handler installed ({e}); a killed run may leave secret material on disk");
    }
}

/// Teardown of per-run secret material at pipeline end (the rounds share one
/// persistence file): the raw `Input-P*-0` streams are always truncated, while a
/// successful `--reuse-extraction` run keeps the slab and its sidecar.
pub(crate) struct SecretsCleanup {
    spdz_path: PathBuf,
    preserve_persistence: Cell<bool>,
}

impl SecretsCleanup {
    pub(crate) fn new(cfg: &SpdzConfig) -> Self {
        // Publish the same state to the signal handler, so a SIGTERM between
        // here and Drop wipes exactly what Drop would wipe.
        *ACTIVE_CLEANUP.lock().unwrap_or_else(|e| e.into_inner()) =
            Some((cfg.spdz_path.clone(), false));
        SecretsCleanup { spdz_path: cfg.spdz_path.clone(), preserve_persistence: Cell::new(false) }
    }

    /// Arm slab survival: on Drop keep `Transactions-P*.data` + the sidecar.
    /// Only for a fully successful reuse run; raw Input streams still go.
    pub(crate) fn preserve_persistence(&self) {
        self.preserve_persistence.set(true);
        if let Some(state) = ACTIVE_CLEANUP.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
            state.1 = true;
        }
    }
}

impl Drop for SecretsCleanup {
    fn drop(&mut self) {
        let preserve = self.preserve_persistence.get();
        let cleaned = teardown_secret_material(&self.spdz_path, preserve);
        *ACTIVE_CLEANUP.lock().unwrap_or_else(|e| e.into_inner()) = None;
        if preserve {
            eprintln!(
                "[SPDZ] Truncated raw inputs + memory dumps: {}; PRESERVED the extract \
                 slab (--reuse-extraction success)",
                if cleaned.is_empty() { "none".into() } else { cleaned.join(", ") }
            );
        } else if !cleaned.is_empty() {
            let mut cleaned = cleaned;
            cleaned.sort();
            eprintln!("[SPDZ] Cleaned per-run secret material: {}", cleaned.join(", "));
        }
    }
}

/// `(n_cases, per-party aligned slice lengths)`, padded the way
/// `write_inputs_nparty` pads, so `--reuse-extraction` can build the identity
/// key without writing inputs. The per-party vectors must be index-aligned
/// (checked here): the writer pads by index, and a missing case would merge
/// into its neighbour's slot.
pub(crate) fn slab_shape(all_cases: &[Vec<CaseData>]) -> std::io::Result<(usize, Vec<usize>)> {
    let align = crate::constants::TRACE_LEN_ALIGNMENT;
    let n_cases = all_cases.first().map_or(0, |c| c.len());
    for (p, cases) in all_cases.iter().enumerate() {
        if cases.len() != n_cases || cases.iter().zip(&all_cases[0]).any(|(a, b)| a.id != b.id) {
            return Err(std::io::Error::other(format!(
                "Party {} case vector is not index-aligned with party 0 ({} vs {} cases): \
                 every party must emit one case per agreed case, sorted by id, empty where \
                 it holds no events (data_loading::prepare_cases_for_mpc)",
                p,
                cases.len(),
                n_cases
            )));
        }
    }
    let partial_lens = all_cases
        .iter()
        .map(|cases| {
            let party_max = cases.iter().map(|c| c.events.len()).max().unwrap_or(1);
            party_max.max(1).div_ceil(align) * align
        })
        .collect();
    Ok((n_cases, partial_lens))
}

/// The epoch-floored `min_ts` every party stream is written relative to. The
/// epoch is a multiple of MS_TO_SECS, so flooring shifts every scaled timestamp
/// by a constant — which holds only for a whole-second min_ts, guarded here.
fn merge_epoch_min_ts(all_cases: &[Vec<CaseData>], dict_size: usize) -> std::io::Result<u64> {
    let min_ts = all_cases
        .iter()
        .flat_map(|cases| cases.iter().flat_map(|c| c.events.iter().map(|(ts, _)| *ts)))
        .filter(|&ts| ts > 0)
        .min()
        .unwrap_or(0);
    if min_ts % crate::constants::MS_TO_SECS != 0 {
        return Err(std::io::Error::other(format!(
            "federation-wide earliest timestamp {} ms is not whole-second-aligned; \
             day-epoch flooring would shift the scaled-second grid and reorder the merge. \
             Re-export the log with whole-second timestamps",
            min_ts
        )));
    }
    let min_ts = min_ts - (min_ts % crate::constants::MIN_TS_EPOCH_MS);

    // The merge tree sorts by the composite key `ts * DICT_SIZE + act`, with
    // pad sentinel PAD_COMPOSITE = 2^40 above every real key. A scaled
    // timestamp breaking that bound would silently misorder the merge.
    let max_ts_scaled = all_cases
        .iter()
        .flat_map(|cases| cases.iter().flat_map(|c| c.events.iter().map(|(ts, _)| *ts)))
        .map(|ts| ts.saturating_sub(min_ts) / crate::constants::MS_TO_SECS)
        .max()
        .unwrap_or(0);
    let max_composite = (max_ts_scaled as u128 + 1) * dict_size as u128;
    if max_composite >= 1u128 << 40 {
        return Err(std::io::Error::other(format!(
            "Composite sort key overflow: max scaled timestamp {} x dict_size {} reaches \
             the PAD_COMPOSITE sentinel 2^40; the time span of the log is too large",
            max_ts_scaled, dict_size
        )));
    }
    Ok(min_ts)
}

/// One party's `Input-P<p>-0` payload: `act ts` pairs, padded to `plen` per
/// case and to `n_cases` cases. The SINGLE producer of those bytes, so
/// `input_payload_digest` hashes exactly what the writer writes.
fn write_party_stream(
    out: &mut impl Write,
    cases: &[CaseData],
    n_cases: usize,
    plen: usize,
    min_ts: u64,
) -> std::io::Result<()> {
    for case in cases {
        for (ts, act) in &case.events {
            // secure_extract.mpc (Round 1) reads per event: activity, timestamp.
            write!(out, "{} {} ", act, ts.saturating_sub(min_ts) / crate::constants::MS_TO_SECS)?;
        }
        // Pad remaining slots with (PAD_ACT=0, PAD_TS=0)
        for _ in case.events.len()..plen {
            write!(out, "0 0 ")?;
        }
    }
    // Dead under the alignment contract of `slab_shape` (cases.len() ==
    // n_cases); kept as belt-and-braces for the index-padding invariant.
    for _ in cases.len()..n_cases {
        for _ in 0..plen {
            write!(out, "0 0 ")?;
        }
    }
    writeln!(out)
}

/// Digest of the encoded party input streams, in party order: the
/// `--reuse-extraction` slab key's only content field, without which two
/// vertical splits of one log at the same `data_dir` collide. Computed from the
/// in-memory cases, since a tail-only run writes no `Input-P*-0`.
pub(crate) fn input_payload_digest(
    all_cases: &[Vec<CaseData>],
    dict_size: usize,
) -> std::io::Result<String> {
    let (n_cases, partial_lens) = slab_shape(all_cases)?;
    let min_ts = merge_epoch_min_ts(all_cases, dict_size)?;
    let mut per_party = Vec::with_capacity(all_cases.len());
    for (p, cases) in all_cases.iter().enumerate() {
        let mut buf: Vec<u8> = Vec::new();
        write_party_stream(&mut buf, cases, n_cases, partial_lens[p], min_ts)?;
        per_party.push(extract_reuse::sha256_hex(&buf));
    }
    Ok(extract_reuse::ordered_digest(per_party))
}

/// Write n-party inputs: `act ts` pairs per party, padded to that party's own
/// aligned max slice length `partial_lens[p]` (output bit-identical).
// LEAKAGE (simulator only): one orchestrator writes Input-P{i}-0 for every
// party from a single `all_cases` it should not have.
pub(crate) fn write_inputs_nparty(
    cfg: &SpdzConfig,
    all_cases: &[Vec<CaseData>],
    dict_size: usize,
) -> std::io::Result<(usize, Vec<usize>)> {
    let dir = cfg.spdz_path.join("Player-Data");
    fs::create_dir_all(&dir)?;

    let (n_cases, partial_lens) = slab_shape(all_cases)?;
    let min_ts = merge_epoch_min_ts(all_cases, dict_size)?;

    for (p, cases) in all_cases.iter().enumerate() {
        let path = dir.join(format!("Input-P{}-0", p));
        let mut f = BufWriter::new(File::create(&path)?);
        write_party_stream(&mut f, cases, n_cases, partial_lens[p], min_ts)?;
        f.flush()?;
    }

    println!(
        "[SPDZ] Wrote {}-party inputs: N_CASES={}, PARTIAL_LEN={}, PARTIAL_LENS={}",
        all_cases.len(),
        n_cases,
        partial_lens.iter().max().unwrap_or(&1),
        partial_lens.iter().map(|l| l.to_string()).collect::<Vec<_>>().join("x")
    );
    Ok((n_cases, partial_lens))
}

/// Optional `PARTIAL_LENS` compile-arg token ("76x8x4x8"), emitted only when the
/// per-party lengths are non-uniform, so a uniform run keeps its compile-arg
/// list, and therefore its content-addressed bytecode, unchanged.
pub(super) fn partial_lens_arg(partial_lens: &[usize]) -> Option<String> {
    let max = *partial_lens.iter().max()?;
    if partial_lens.iter().all(|&l| l == max) {
        None
    } else {
        Some(partial_lens.iter().map(|l| l.to_string()).collect::<Vec<_>>().join("x"))
    }
}
