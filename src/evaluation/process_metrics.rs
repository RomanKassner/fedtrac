//! Process mining metrics via Python bridge.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use crate::constants::EVAL_TIMEOUT_SECS;
use crate::error::{BridgeError, Result};

fn python_path() -> &'static str {
    if Path::new(".venv/bin/python").exists() {
        ".venv/bin/python"
    } else {
        "python3"
    }
}

fn run_python(script: &Path, build_args: impl FnOnce(&mut Command)) -> Result<Output> {
    if !script.exists() {
        return Err(BridgeError::NotFound(script.to_path_buf()).into());
    }

    let mut cmd = Command::new(python_path());
    cmd.arg(script);
    build_args(&mut cmd);

    let output = cmd.output()?;

    if !output.status.success() {
        return Err(BridgeError::Subprocess {
            script: script.display().to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        }
        .into());
    }
    Ok(output)
}

// `python -m`: needed for bridge/ modules that use relative imports.
fn run_python_module(module: &str, build_args: impl FnOnce(&mut Command)) -> Result<Output> {
    let mut cmd = Command::new(python_path());
    cmd.arg("-m").arg(module);
    build_args(&mut cmd);

    let output = cmd.output()?;

    if !output.status.success() {
        return Err(BridgeError::Subprocess {
            script: module.to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        }
        .into());
    }
    Ok(output)
}

// Like `run_python_module` but kills the child after `timeout`: the conformance
// eval's token-replay step can hang indefinitely on a pathological model.
fn run_python_module_timeout(
    module: &str,
    timeout: Duration,
    build_args: impl FnOnce(&mut Command),
) -> Result<Output> {
    let mut cmd = Command::new(python_path());
    cmd.arg("-m").arg(module).stdout(Stdio::piped()).stderr(Stdio::piped());
    build_args(&mut cmd);
    let mut child = cmd.spawn()?;

    // Drain both pipes in threads, else the child blocks once output exceeds
    // the OS pipe buffer (~64 KB).
    let mut out_pipe = child.stdout.take().unwrap();
    let mut err_pipe = child.stderr.take().unwrap();
    let out_h = thread::spawn(move || {
        let mut b = Vec::new();
        let _ = out_pipe.read_to_end(&mut b);
        b
    });
    let err_h = thread::spawn(move || {
        let mut b = Vec::new();
        let _ = err_pipe.read_to_end(&mut b);
        b
    });

    let start = Instant::now();
    loop {
        match child.try_wait()? {
            Some(status) => {
                let stdout = out_h.join().unwrap_or_default();
                let stderr = err_h.join().unwrap_or_default();
                if !status.success() {
                    return Err(BridgeError::Subprocess {
                        script: module.to_string(),
                        stderr: String::from_utf8_lossy(&stderr).into_owned(),
                    }
                    .into());
                }
                return Ok(Output { status, stdout, stderr });
            }
            None if start.elapsed() >= timeout => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(BridgeError::Subprocess {
                    script: module.to_string(),
                    stderr: format!(
                        "timed out after {}s and was killed; MPC artifacts are saved \
                         -- re-run `{}` on the run dir to recover metrics",
                        timeout.as_secs(),
                        module
                    ),
                }
                .into());
            }
            None => thread::sleep(Duration::from_millis(200)),
        }
    }
}

fn forward_stdout(output: &Output, tag: &str) {
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if !line.trim().is_empty() {
            println!("[{}] {}", tag, line);
        }
    }
}

pub fn run_evaluate_all(output_dir: &Path, tier: &str, fast_eval: bool) -> Result<()> {
    let output = run_python_module_timeout(
        "bridge.evaluate",
        Duration::from_secs(EVAL_TIMEOUT_SECS),
        |cmd| {
            cmd.arg(output_dir).arg("--tier").arg(tier);
            if fast_eval {
                cmd.arg("--skip-log-baseline").arg("--token-replay");
            }
        },
    )?;
    forward_stdout(&output, "Eval");
    Ok(())
}

pub struct ClusterDispatchOutput {
    pub assignments: Vec<usize>,
    pub k_actual: usize,
}

/// `bridge.cluster --algo sequence`: Markov-chain hard-EM on the dense d² rows,
/// seeded with the k-means partition.
pub fn run_cluster_dispatch_sequence(
    rows_csv: &Path,
    k: usize,
    dict_size: usize,
    max_iter: usize,
    init_labels: &Path,
    n_rows: usize,
) -> Result<ClusterDispatchOutput> {
    let output = run_python_module("bridge.cluster", |cmd| {
        cmd.arg("--features")
            .arg(rows_csv)
            .arg("--algo")
            .arg("sequence")
            .arg("--k")
            .arg(k.to_string())
            .arg("--dict-size")
            .arg(dict_size.to_string())
            .arg("--max-iter")
            .arg(max_iter.to_string());
        cmd.arg("--init-labels").arg(init_labels);
    })?;
    parse_cluster_dispatch(&output.stdout, Some(n_rows))
}

/// `expect_rows` is the number of feature rows sent in: a dropped or extra
/// label would be zipped positionally against the traces in `export.rs`,
/// shifting later cases into a neighbour's cluster — so both are hard errors.
fn parse_cluster_dispatch(
    stdout_bytes: &[u8],
    expect_rows: Option<usize>,
) -> Result<ClusterDispatchOutput> {
    let stdout = String::from_utf8_lossy(stdout_bytes);
    let parsed: serde_json::Value =
        serde_json::from_str(stdout.trim()).map_err(|e| BridgeError::Serde {
            script: "bridge.cluster".to_string(),
            detail: format!("JSON parse: {}", e),
        })?;

    let assignments: Vec<usize> = parsed["assignments"]
        .as_array()
        .ok_or_else(|| BridgeError::Serde {
            script: "bridge.cluster".to_string(),
            detail: "missing field `assignments`".into(),
        })?
        .iter()
        .map(|v| {
            v.as_u64().map(|n| n as usize).ok_or_else(|| BridgeError::Serde {
                script: "bridge.cluster".to_string(),
                detail: format!("assignment {} is not a non-negative integer", v),
            })
        })
        .collect::<std::result::Result<Vec<usize>, BridgeError>>()?;
    if let Some(n) = expect_rows {
        if assignments.len() != n {
            return Err(BridgeError::Serde {
                script: "bridge.cluster".to_string(),
                detail: format!("{} assignments for {} feature rows", assignments.len(), n),
            }
            .into());
        }
    }

    let k_actual = parsed["k_actual"]
        .as_u64()
        .map(|v| v as usize)
        .unwrap_or_else(|| assignments.iter().copied().max().map_or(0, |m| m + 1));

    Ok(ClusterDispatchOutput { assignments, k_actual })
}

pub fn run_k_select(features_csv: &Path, k_min: usize, trace_len: Option<usize>) -> Result<usize> {
    let script = Path::new("scripts/k_select.py");
    let output = run_python(script, |cmd| {
        cmd.arg(features_csv).arg("--format").arg("json").arg("--k-min").arg(k_min.to_string());
        if let Some(tl) = trace_len {
            cmd.arg("--trace-len").arg(tl.to_string());
        }
    })?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value =
        serde_json::from_str(stdout.trim()).map_err(|e| BridgeError::Serde {
            script: "scripts/k_select.py".to_string(),
            detail: format!("JSON parse: {}", e),
        })?;

    let k = parsed["recommended_k"].as_u64().ok_or_else(|| BridgeError::Serde {
        script: "scripts/k_select.py".to_string(),
        detail: "missing field `recommended_k`".into(),
    })? as usize;

    let method = parsed["method"].as_str().unwrap_or("unknown");
    let ps = parsed["ps"].as_f64().map(|v| format!("{:.3}", v)).unwrap_or_default();
    println!("[Auto-k] k={} (method={}, PS={})", k, method, ps);

    Ok(k)
}
