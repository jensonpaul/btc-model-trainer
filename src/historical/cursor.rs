//! Persistent per-exchange fetch cursor.
//!
//! The cursor records the timestamp of the **last successfully written row**
//! for an exchange.  On restart, each fetcher resumes from `cursor + 1 µs`
//! instead of re-downloading data that is already on disk.
//!
//! # File location
//!
//! `{data_root}/{exchange}/cursor.json`
//!
//! # Format
//!
//! ```json
//! {
//!   "last_ts_micros": 1735689600000000,
//!   "last_ts_human":  "2025-01-01T00:00:00Z",
//!   "rows_written":   12345678
//! }
//! ```

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cursor {
    /// µs since UNIX epoch of the last row committed to disk.
    /// `None` means no data has been written yet — start from the exchange
    /// floor date.
    pub last_ts_micros: Option<i64>,

    /// Human-readable form of `last_ts_micros` (written for debuggability,
    /// not read back — the µs value is canonical).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_ts_human: Option<String>,

    /// Cumulative rows written across all runs.
    pub rows_written: u64,
}

impl Cursor {
    /// Load an existing cursor file, or return a blank cursor if the file
    /// does not exist yet.
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::blank());
        }
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading cursor {}", path.display()))?;
        serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing cursor {}", path.display()))
    }

    /// Persist the cursor atomically (write to a `.tmp` file then rename).
    pub fn save(&self, path: &Path) -> Result<()> {
        let tmp = path.with_extension("json.tmp");
        let bytes = serde_json::to_vec_pretty(self).context("serialising cursor")?;
        std::fs::write(&tmp, &bytes)
            .with_context(|| format!("writing cursor tmp {}", tmp.display()))?;
        std::fs::rename(&tmp, path)
            .with_context(|| format!("renaming cursor tmp → {}", path.display()))?;
        Ok(())
    }

    /// Advance the cursor after a successful batch write.
    pub fn advance(&mut self, last_ts_micros: i64, rows: u64) {
        let dt: DateTime<Utc> = Utc
            .timestamp_micros(last_ts_micros)
            .single()
            .unwrap_or(DateTime::<Utc>::MIN_UTC);
        self.last_ts_micros = Some(last_ts_micros);
        self.last_ts_human  = Some(dt.format("%Y-%m-%dT%H:%M:%SZ").to_string());
        self.rows_written   += rows;
    }

    fn blank() -> Self {
        Self { last_ts_micros: None, last_ts_human: None, rows_written: 0 }
    }
}

/// Canonical path for a cursor file given the data root and exchange name.
pub fn cursor_path(data_root: &Path, exchange: &str) -> PathBuf {
    data_root.join(exchange).join("cursor.json")
}
