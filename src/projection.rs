//! Achlioptas sparse random projection for JL dimensionality reduction.

use std::io::Write;
use std::path::Path;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Sparse random projection with entries in {-1, 0, +1} at {1/6, 2/3, 1/6}.
pub struct AchlioptasMatrix {
    pub data: Vec<Vec<i8>>,
    pub k_proj: usize,
    pub d_original: usize,
}

impl AchlioptasMatrix {
    /// Passthrough for an encoding that already fits below the target dim.
    pub fn identity(d: usize) -> Self {
        let mut data = vec![vec![0i8; d]; d];
        for (i, row) in data.iter_mut().enumerate() {
            row[i] = 1;
        }
        Self { data, k_proj: d, d_original: d }
    }

    /// Generate from a deterministic seed.
    pub fn generate(k_proj: usize, d_original: usize, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut data = vec![vec![0i8; d_original]; k_proj];
        for row in &mut data {
            for cell in row.iter_mut() {
                let coin: f64 = rng.gen();
                *cell = if coin < 1.0 / 6.0 {
                    -1
                } else if coin > 5.0 / 6.0 {
                    1
                } else {
                    0
                }
            }
        }
        Self { data, k_proj, d_original }
    }

    /// `y[j] = Σ_l R[j][col_map(l)]·x[l]`; `col_map` densifies sparse-indexed
    /// feature columns (`None` = already dense over `d_original`).
    pub fn project(&self, features: &[Vec<f64>], col_map: Option<&[usize]>) -> Vec<Vec<f64>> {
        features
            .iter()
            .map(|row| {
                let mut y = vec![0.0f64; self.k_proj];
                for (l, &v) in row.iter().enumerate() {
                    if v == 0.0 {
                        continue;
                    }
                    let dense = col_map.map_or(l, |m| m[l]);
                    for (j, out) in y.iter_mut().enumerate() {
                        match self.data[j][dense] {
                            1 => *out += v,
                            -1 => *out -= v,
                            _ => {}
                        }
                    }
                }
                y
            })
            .collect()
    }

    /// Dimension header, then one row per line, for the MPC program.
    pub fn write_to_file(&self, path: &Path) -> std::io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        writeln!(f, "{} {}", self.k_proj, self.d_original)?;
        for row in &self.data {
            let line: Vec<String> = row.iter().map(|v| v.to_string()).collect();
            writeln!(f, "{}", line.join(" "))?;
        }
        Ok(())
    }
}
