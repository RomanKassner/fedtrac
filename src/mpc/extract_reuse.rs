//! Extract-once / branch-many amortization (`--reuse-extraction`): Round 1's
//! RNG-free core (input read + merge + extract) is persisted once with an
//! identity sidecar; a later run whose key matches EXACTLY reads it instead.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::SpdzError;

/// How Round 1 should treat the persistence slab for a given run.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ExtractReuse {
    /// Non-reuse: write inputs, full extract, persist nothing extra.
    None,
    /// First matching reuse run: full extract that ALSO persists the encoding
    /// slab at `enc_offset` (`PERSIST_ENC=1`).
    PersistCore { enc_offset: usize },
    /// Later matching reuse run: skip Steps 1-2, read the slab (`CORE_MODE=2`).
    TailOnly { enc_offset: usize, bigram_offset: usize },
}

impl ExtractReuse {
    pub(crate) fn is_tail_only(&self) -> bool {
        matches!(self, ExtractReuse::TailOnly { .. })
    }

    /// The encoding-slab offset when this run participates in reuse (`None` for
    /// a plain run). Callers assert it clears the resolved mode layout end.
    pub(crate) fn enc_offset(&self) -> Option<usize> {
        match self {
            ExtractReuse::None => None,
            ExtractReuse::PersistCore { enc_offset }
            | ExtractReuse::TailOnly { enc_offset, .. } => Some(*enc_offset),
        }
    }

    /// `(CORE_MODE, PERSIST_ENC, ENC_OFFSET, BIGRAM_OFFSET)` for the extract
    /// compile args, or `None` for a plain (non-reuse) run whose arg list must
    /// stay byte-identical to today's.
    pub(crate) fn compile_tuple(&self) -> Option<(u8, u8, usize, usize)> {
        match self {
            ExtractReuse::None => None,
            ExtractReuse::PersistCore { enc_offset } => Some((0, 1, *enc_offset, 0)),
            ExtractReuse::TailOnly { enc_offset, bigram_offset } => {
                Some((2, 0, *enc_offset, *bigram_offset))
            }
        }
    }
}

/// Slab identity: two runs with an identical key produce a bit-identical core
/// slab. K, k_proj and the projection matrix are deliberately absent — the slab
/// is independent of all three — and every other field closes a silent
/// wrong-slab reuse.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct SlabKey {
    pub data_dir: String,
    /// sha256 over the resolved, ordered case-id list actually fed to Round 1.
    pub case_set_sha256: String,
    /// sha256 over the ordered activity dictionary (its ids ARE the encoding).
    pub dictionary_sha256: String,
    /// sha256 over the encoded party input streams, in party order: two vertical
    /// splits of one log, or a timestamp re-export, agree on every other field
    /// and still produce a different slab.
    pub input_payload_sha256: String,
    pub n_cases: usize,
    /// The truncation L this run applied per party slice, before alignment:
    /// `partial_lens` alone cannot separate two L's that round to one bound.
    pub max_trace_len_effective: usize,
    pub partial_lens: Vec<usize>,
    pub dict_size: usize,
    pub encoding: String,
    pub enc_dim: usize,
    pub protocol: String,
    /// compile.py `-R <ring_size>` and the extra `CLUSTERING_MPC_COMPILE_FLAGS`:
    /// both change the share representation the slab is written in.
    pub ring_size: u32,
    pub compile_flags: Vec<String>,
    pub n_parties: usize,
}

impl SlabKey {
    /// Field-by-field diff for the FAIL-LOUD mismatch message.
    fn diff(&self, other: &SlabKey) -> Vec<String> {
        let mut d = Vec::new();
        macro_rules! cmp {
            ($f:ident) => {
                if self.$f != other.$f {
                    d.push(format!(
                        "  {}: sidecar={:?} run={:?}",
                        stringify!($f),
                        self.$f,
                        other.$f
                    ));
                }
            };
        }
        cmp!(data_dir);
        cmp!(case_set_sha256);
        cmp!(dictionary_sha256);
        cmp!(input_payload_sha256);
        cmp!(n_cases);
        cmp!(max_trace_len_effective);
        cmp!(partial_lens);
        cmp!(dict_size);
        cmp!(encoding);
        cmp!(enc_dim);
        cmp!(protocol);
        cmp!(ring_size);
        cmp!(compile_flags);
        cmp!(n_parties);
        d
    }
}

/// Sidecar file content: the identity key plus the two Rust-supplied offsets
/// the tail run reads the slab from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SlabSidecar {
    pub key: SlabKey,
    pub bigram_offset: usize,
    pub enc_offset: usize,
}

/// Largest cluster count a `--reuse-extraction` run may resolve: the encoding
/// slab leaves room for this many label columns, and a larger k fails loud at
/// the `mode_layout_end_bound` check of its mode.
pub(crate) const SLAB_MAX_CLUSTERS: usize = 64;

/// Where the encoding slab rides: above every mode's persistence write range.
/// The mode slabs end at most at `n*d² + n*enc_dim + 2n*(K+1)` (K_PROJ <=
/// enc_dim) with K capped at `SLAB_MAX_CLUSTERS`, which keeps the zero-filled
/// gap linear in N. Depends only on slab-key fields, so every branch run
/// computes the same offset the full run wrote at.
pub(crate) fn enc_slab_offset(n_cases: usize, d_sq: usize, enc_dim: usize) -> usize {
    n_cases * d_sq + n_cases * enc_dim + 2 * n_cases * (SLAB_MAX_CLUSTERS + 1)
}

/// Highest persistence position any mode writes for the resolved `(k_proj, k)`.
/// The orchestrator asserts this is `<= enc_slab_offset(..)` once k is known.
pub(crate) fn mode_layout_end_bound(n_cases: usize, d_sq: usize, k_proj: usize, k: usize) -> usize {
    // SK: n*d² + n*k_proj + n*k ; rsec: n*d² + 2n + n*k_proj + n*k. The
    // union bound below dominates both (with the retired refine slab's slack).
    n_cases * d_sq + 2 * n_cases + n_cases * k_proj + 2 * n_cases * k
}

pub(crate) fn sidecar_path(spdz_path: &Path) -> PathBuf {
    spdz_path.join("Player-Data").join("extract_slab.json")
}

/// sha256 over an ordered list of strings, length-prefixed per item so no
/// separator can be forged by a value that contains one.
pub(crate) fn ordered_digest(items: impl IntoIterator<Item = impl AsRef<str>>) -> String {
    let mut buf: Vec<u8> = Vec::new();
    for item in items {
        let s = item.as_ref();
        buf.extend_from_slice(format!("{}:", s.len()).as_bytes());
        buf.extend_from_slice(s.as_bytes());
        buf.push(b'\n');
    }
    sha256_hex(&buf)
}

/// Decide the reuse mode for a run and fail loud on a present-but-divergent
/// sidecar. A matching sidecar's own offsets are where the slab lives.
pub(crate) fn resolve(
    spdz_path: &Path,
    key: &SlabKey,
    enc_offset: usize,
) -> Result<ExtractReuse, SpdzError> {
    match read_sidecar(spdz_path) {
        None => Ok(ExtractReuse::PersistCore { enc_offset }),
        Some(sc) if sc.key == *key => Ok(ExtractReuse::TailOnly {
            enc_offset: sc.enc_offset,
            bigram_offset: sc.bigram_offset,
        }),
        Some(sc) => {
            let diffs = sc.key.diff(key);
            Err(SpdzError::Runtime(format!(
                "--reuse-extraction: {} does not match this run's slab; refusing to \
                 re-extract over a foreign slab. Differences (sidecar vs run):\n{}\n\
                 Remove the sidecar (and the stale Persistence) or run without \
                 --reuse-extraction to regenerate.",
                sidecar_path(spdz_path).display(),
                if diffs.is_empty() {
                    "  (identity fields equal but comparison failed)".to_string()
                } else {
                    diffs.join("\n")
                }
            )))
        }
    }
}

fn read_sidecar(spdz_path: &Path) -> Option<SlabSidecar> {
    let bytes = std::fs::read(sidecar_path(spdz_path)).ok()?;
    match serde_json::from_slice(&bytes) {
        Ok(sc) => Some(sc),
        Err(e) => {
            // An unreadable sidecar means a full re-extract, the safe side.
            eprintln!(
                "[MPC] WARNING: {} is unreadable ({}); treating it as absent and \
                 re-extracting the core slab",
                sidecar_path(spdz_path).display(),
                e
            );
            None
        }
    }
}

/// Persist the sidecar next to the slab, on a successful reuse run only.
pub(crate) fn write_sidecar(
    spdz_path: &Path,
    key: &SlabKey,
    bigram_offset: usize,
    enc_offset: usize,
) -> Result<(), SpdzError> {
    let sc = SlabSidecar { key: key.clone(), bigram_offset, enc_offset };
    let s = serde_json::to_string_pretty(&sc).map_err(|e| SpdzError::Runtime(e.to_string()))?;
    let path = sidecar_path(spdz_path);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, s)?;
    Ok(())
}

// Minimal vendored SHA-256, verified against `sha256sum` in tests.

pub(crate) fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, wi) in w.iter_mut().enumerate().take(16) {
            *wi = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ ((!v[4]) & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v[7] = v[6];
            v[6] = v[5];
            v[5] = v[4];
            v[4] = v[3].wrapping_add(t1);
            v[3] = v[2];
            v[2] = v[1];
            v[1] = v[0];
            v[0] = t1.wrapping_add(t2);
        }
        for (hi, vi) in h.iter_mut().zip(v.iter()) {
            *hi = hi.wrapping_add(*vi);
        }
    }

    let mut out = String::with_capacity(64);
    for word in h {
        out.push_str(&format!("{:08x}", word));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn enc_slab_offset_clears_mode_layout() {
        // The encoding slab must ride above every mode's write range, for all
        // k_proj <= enc_dim and k <= SLAB_MAX_CLUSTERS.
        for &(n, d_sq, enc_dim) in &[(500usize, 100usize, 100usize), (20, 36, 216), (8, 64, 64)] {
            let off = enc_slab_offset(n, d_sq, enc_dim);
            for k_proj in [1, enc_dim] {
                for k in [1, SLAB_MAX_CLUSTERS] {
                    assert!(
                        mode_layout_end_bound(n, d_sq, k_proj.min(enc_dim), k) <= off,
                        "n={n} d_sq={d_sq} enc_dim={enc_dim} k_proj={k_proj} k={k}"
                    );
                }
            }
            // One cluster past the cap is NOT cleared: the mode's own check refuses it.
            assert!(mode_layout_end_bound(n, d_sq, enc_dim, SLAB_MAX_CLUSTERS + 1) > off);
        }
        // Linear in N: full-N RTFM under trigram (d² = 196, enc_dim = 14³)
        // stays below 2^29 slots.
        assert!(enc_slab_offset(129_985, 196, 2744) < 1 << 29);
    }

    fn test_key() -> SlabKey {
        SlabKey {
            data_dir: "data/splits/x".into(),
            case_set_sha256: ordered_digest(["c1", "c2"]),
            dictionary_sha256: ordered_digest(["A", "B", "C"]),
            input_payload_sha256: ordered_digest(["3 0 4 1 ", "3 0 0 0 "]),
            n_cases: 10,
            max_trace_len_effective: 12,
            partial_lens: vec![4, 4],
            dict_size: 6,
            encoding: "Bigram".into(),
            enc_dim: 36,
            protocol: "semi2k-party.x".into(),
            ring_size: 64,
            compile_flags: vec!["-Z".into(), "2".into()],
            n_parties: 2,
        }
    }

    #[test]
    fn ordered_digest_separates_reorderings_and_boundaries() {
        // Order matters (the ids are index-aligned with the party streams) and
        // the length prefix stops "ab","c" from colliding with "a","bc".
        assert_ne!(ordered_digest(["a", "b"]), ordered_digest(["b", "a"]));
        assert_ne!(ordered_digest(["ab", "c"]), ordered_digest(["a", "bc"]));
        assert_eq!(ordered_digest(["a", "b"]), ordered_digest(["a", "b"]));
        // Same digest whether the caller holds Strings or &str.
        assert_eq!(ordered_digest(["a"]), ordered_digest(&[String::from("a")]));
    }

    /// Each widened key field must, on its own, make a sidecar refuse to match.
    #[test]
    fn widened_key_fields_each_refuse_reuse() {
        let dir = std::env::temp_dir().join(format!("extract_reuse_neg_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("Player-Data")).unwrap();
        let key = test_key();
        let enc_off = enc_slab_offset(10, 36, 36);
        write_sidecar(&dir, &key, 0, enc_off).unwrap();

        // L=10 and L=12 both align to partial_lens 12 at TRACE_LEN_ALIGNMENT=4,
        // so only max_trace_len_effective separates them.
        let mut l10 = key.clone();
        l10.max_trace_len_effective = 10;
        assert_eq!(l10.partial_lens, key.partial_lens);
        // A -R 128 rebuild over a -R 64 slab.
        let mut r128 = key.clone();
        r128.ring_size = 128;
        let mut flags = key.clone();
        flags.compile_flags = vec![];
        let mut cases = key.clone();
        cases.case_set_sha256 = ordered_digest(["c1", "c3"]);
        let mut dict = key.clone();
        dict.dictionary_sha256 = ordered_digest(["A", "B", "D"]);
        // Same cases, dictionary and shapes: only one event's timestamp (or
        // which party holds it) moved. That is a different slab.
        let mut payload = key.clone();
        payload.input_payload_sha256 = ordered_digest(["3 0 4 2 ", "3 0 0 0 "]);

        for (label, bad) in [
            ("max_trace_len_effective", l10),
            ("ring_size", r128),
            ("compile_flags", flags),
            ("case_set_sha256", cases),
            ("dictionary_sha256", dict),
            ("input_payload_sha256", payload),
        ] {
            let err = resolve(&dir, &bad, enc_off)
                .expect_err(&format!("{} must not reuse the slab", label));
            let msg = format!("{}", err);
            assert!(msg.contains("does not match"), "{}: {}", label, msg);
            assert!(msg.contains(label), "mismatch message must name {}: {}", label, msg);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tail_only_matches_sidecar_key() {
        let dir = std::env::temp_dir().join(format!("extract_reuse_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("Player-Data")).unwrap();
        let key = test_key();
        let enc_off = enc_slab_offset(10, 36, 36);
        // Absent sidecar -> PersistCore.
        assert!(matches!(resolve(&dir, &key, enc_off).unwrap(), ExtractReuse::PersistCore { .. }));
        write_sidecar(&dir, &key, 0, enc_off).unwrap();
        // Present + matching -> TailOnly at the recorded offsets.
        match resolve(&dir, &key, enc_off).unwrap() {
            ExtractReuse::TailOnly { enc_offset, bigram_offset } => {
                assert_eq!(enc_offset, enc_off);
                assert_eq!(bigram_offset, 0);
            }
            other => panic!("expected TailOnly, got {:?}", other),
        }
        // Present + divergent -> FAIL LOUD.
        let mut bad = key.clone();
        bad.n_cases = 11;
        let err = resolve(&dir, &bad, enc_off).unwrap_err();
        assert!(format!("{}", err).contains("does not match"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
