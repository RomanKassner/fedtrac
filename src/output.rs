//! Auto-timestamped run directories.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::error::{PipelineError, Result};

const RUNS_DIR: &str = "output/runs";

/// Auto-timestamped run directory: `output/runs/[<name>_]<timestamp>/`.
pub fn resolve_output_dir(run_name: &Option<String>) -> PathBuf {
    let ts = timestamp_suffix();
    let name = match run_name {
        Some(n) => format!("{}_{}", n, ts),
        None => ts,
    };
    PathBuf::from(RUNS_DIR).join(name)
}

pub fn clean_output_dir(dir: &Path) -> Result<()> {
    if dir.exists() {
        fs::remove_dir_all(dir).map_err(PipelineError::Io)?;
    }
    fs::create_dir_all(dir).map_err(PipelineError::Io)?;
    Ok(())
}

fn timestamp_suffix() -> String {
    let secs =
        SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().as_secs();

    let days = secs / 86400;
    let time_of_day = secs % 86400;

    let (year, month, day) = days_to_ymd(days);
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    format!("{:04}-{:02}-{:02}_{:02}-{:02}-{:02}", year, month, day, hours, minutes, seconds)
}

fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    // Civil calendar algorithm (Howard Hinnant).
    days += 719_468;
    let era = days / 146_097;
    let doe = days % 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}
