//! Pipeline-wide error types.

use std::fmt;
use std::path::PathBuf;

use crate::mpc::SpdzError;

#[derive(Debug)]
pub enum PipelineError {
    Io(std::io::Error),
    Mpc(String),
    Data(String),
    Clustering(String),
    Bridge(BridgeError),
}

/// Failures in the Python bridge, split by layer.
#[derive(Debug)]
pub enum BridgeError {
    /// Script file not present on disk.
    NotFound(PathBuf),
    /// Subprocess exited non-zero.
    Subprocess { script: String, stderr: String },
    /// Script exited zero but its output was malformed.
    Serde { script: String, detail: String },
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PipelineError::Io(e) => write!(f, "IO error: {}", e),
            PipelineError::Mpc(s) => write!(f, "MPC error: {}", s),
            PipelineError::Data(s) => write!(f, "Data error: {}", s),
            PipelineError::Clustering(s) => write!(f, "Clustering error: {}", s),
            PipelineError::Bridge(e) => write!(f, "Bridge error: {}", e),
        }
    }
}

impl fmt::Display for BridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BridgeError::NotFound(p) => {
                write!(f, "{} not found (run from project root)", p.display())
            }
            BridgeError::Subprocess { script, stderr } => {
                write!(f, "{} failed: {}", script, stderr)
            }
            BridgeError::Serde { script, detail } => {
                write!(f, "{} returned invalid output: {}", script, detail)
            }
        }
    }
}

impl std::error::Error for PipelineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PipelineError::Io(e) => Some(e),
            PipelineError::Bridge(e) => Some(e),
            _ => None,
        }
    }
}

impl std::error::Error for BridgeError {}

impl From<std::io::Error> for PipelineError {
    fn from(e: std::io::Error) -> Self {
        PipelineError::Io(e)
    }
}

impl From<SpdzError> for PipelineError {
    fn from(e: SpdzError) -> Self {
        PipelineError::Mpc(e.to_string())
    }
}

impl From<BridgeError> for PipelineError {
    fn from(e: BridgeError) -> Self {
        PipelineError::Bridge(e)
    }
}

pub type Result<T> = std::result::Result<T, PipelineError>;
