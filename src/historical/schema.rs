//! Canonical Arrow schema for on-disk trade row storage.
//!
//! Every source normalises its raw API response into [`TradeRow`] before
//! writing.  The Parquet schema is intentionally minimal — the 5 columns
//! exactly match the CSV format that `src/main.rs` already expects, so
//! reading parquet → emitting CSV requires no transformation beyond
//! column extraction.

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};

/// Shared Arrow schema used for every `*.parquet` shard.
///
/// Column order is fixed; callers must not reorder fields when building
/// record batches.
pub static SCHEMA: std::sync::LazyLock<Arc<Schema>> = std::sync::LazyLock::new(|| {
    Arc::new(Schema::new(vec![
        Field::new("ts_micros", DataType::Int64,  false),
        Field::new("price",     DataType::Float64, false),
        Field::new("quantity",  DataType::Float64, false),
        // nullable: not all sources provide taker side
        Field::new("side",      DataType::Utf8,    true),
        Field::new("exchange",  DataType::Utf8,    false),
    ]))
});

/// In-memory representation of one normalised trade row.
///
/// All sources map their native fields to this type before batching.
#[derive(Debug, Clone)]
pub struct TradeRow {
    /// Microseconds since UNIX epoch, **ascending**.
    pub ts_micros: i64,
    /// Executed price in USD.
    pub price:     f64,
    /// Executed quantity in BTC.
    pub quantity:  f64,
    /// Taker side.  `None` when the source does not expose it (e.g. Kraken
    /// OHLC and Coinbase candles — aggregated bars have no individual side).
    pub side:      Option<&'static str>,  // "buy" | "sell" | None
    /// Source exchange name, lower-case ASCII.
    pub exchange:  &'static str,
}
