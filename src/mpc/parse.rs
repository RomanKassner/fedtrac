//! Parsers for MP-SPDZ party-0 stdout.

use super::SpdzError;

/// Parse `ROW_START / B i v / ROW_END` blocks; both `-n` and the `u64 >= 2^63`
/// two's-complement wrap are accepted.
pub(crate) fn parse_row_output(
    output: &str,
    expected_rows: usize,
    row_width: usize,
) -> Result<Vec<Vec<u64>>, SpdzError> {
    let mut rows: Vec<Vec<u64>> = Vec::new();
    let mut current: Option<Vec<u64>> = None;
    let mut current_seen: Vec<bool> = Vec::new();

    for line in output.lines() {
        if line.starts_with("ROW_START") {
            current = Some(vec![0u64; row_width]);
            current_seen = vec![false; row_width];
        } else if line.starts_with("ROW_END") {
            if let Some(row) = current.take() {
                // A dropped stdout line must not silently zero a revealed cell.
                let missing = current_seen.iter().filter(|&&s| !s).count();
                if missing > 0 {
                    return Err(SpdzError::Runtime(format!(
                        "Row {}: {}/{} cells missing in stdout",
                        rows.len(),
                        missing,
                        row_width
                    )));
                }
                rows.push(row);
            }
        } else if let Some(ref mut row) = current {
            if let Some(rest) = line.strip_prefix("B ") {
                let parts: Vec<_> = rest.split_whitespace().collect();
                if parts.len() == 2 {
                    if let Ok(b) = parts[0].parse::<usize>() {
                        let v = parts[1]
                            .parse::<i64>()
                            .ok()
                            .map(|v| v as u64)
                            .or_else(|| parts[1].parse::<u64>().ok());
                        if let (Some(v), true) = (v, b < row_width) {
                            row[b] = v;
                            current_seen[b] = true;
                        }
                    }
                }
            }
        }
    }

    if rows.len() == expected_rows {
        Ok(rows)
    } else {
        Err(SpdzError::Runtime(format!("Expected {} rows, got {}", expected_rows, rows.len())))
    }
}

/// Like [`parse_row_output`], deriving each row's backbone flag from the zero
/// mask: suppressed rows open as exact zeros, a backbone row is unit-norm. A
/// mismatch against the opened `N_BACKBONE` count is a degenerate reveal,
/// returned typed so the caller re-draws R instead of silently demoting a
/// backbone row into the hybrid tail.
pub(crate) fn parse_row_output_backbone(
    output: &str,
    expected_rows: usize,
    row_width: usize,
) -> Result<(Vec<Vec<u64>>, Vec<bool>), SpdzError> {
    let rows = parse_row_output(output, expected_rows, row_width)?;
    let backbone: Vec<bool> = rows.iter().map(|r| r.iter().any(|&v| v != 0)).collect();

    let n_backbone = output
        .lines()
        .find_map(|l| l.trim().strip_prefix("N_BACKBONE "))
        .and_then(|v| v.trim().parse::<usize>().ok())
        .ok_or_else(|| SpdzError::Runtime("N_BACKBONE line missing from rsecret stdout".into()))?;
    let n_nonzero = backbone.iter().filter(|&&b| b).count();
    if n_nonzero != n_backbone {
        return Err(SpdzError::DegenerateReveal { n_nonzero, n_backbone });
    }
    Ok((rows, backbone))
}
