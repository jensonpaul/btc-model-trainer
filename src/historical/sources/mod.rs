//! Exchange-specific historical data fetchers.
//!
//! Each fetcher implements [`HistoricalFetcher`] and is responsible for:
//!
//! 1. Reading the exchange's [`Cursor`] to determine where to resume.
//! 2. Paginating through the exchange's REST API from that point forward.
//! 3. Normalising each response into [`TradeRow`]s.
//! 4. Writing complete months to the [`DataStore`] and advancing the cursor.
//!
//! All fetchers are synchronous at the month level — they block until a
//! full month is fetched and written before advancing.  This keeps the
//! on-disk state always consistent: a crashed run leaves at most one
//! partial month, which will be completed on the next run.

pub mod binance;
pub mod kraken;
pub mod bitstamp;
pub mod coinbase;

use anyhow::Result;
use async_trait::async_trait;

/// Common interface for all exchange historical fetchers.
#[async_trait]
pub trait HistoricalFetcher: Send + Sync {
    /// Human-readable exchange name (lower-case ASCII).
    fn exchange(&self) -> &'static str;

    /// Fetch all data from `resume_from_micros` up to `up_to_micros`
    /// (exclusive), writing each completed month to disk and advancing the
    /// cursor.  Both timestamps are µs since UNIX epoch.
    ///
    /// Implementations must be **idempotent**: calling this twice for the
    /// same range must not duplicate rows (the store uses dedup on append).
    async fn fetch_range(
        &self,
        resume_from_micros: i64,
        up_to_micros:       i64,
    ) -> Result<u64>; // returns total rows written
}

/// Shared HTTP client configuration used by all fetchers.
pub fn build_client() -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent("btc-model-trainer/0.2 (historical data collection)")
        .connect_timeout(std::time::Duration::from_secs(30))
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .expect("failed to build HTTP client")
}

/// Polite inter-request delay to stay within public rate limits.
pub async fn rate_limit_sleep(exchange: &str) {
    let ms = match exchange {
        "binance"  => 200,   // Binance Vision: no hard limit, be polite
        "kraken"   => 1_000, // Kraken public: ~1 req/s recommended
        "bitstamp" => 400,   // Bitstamp public: ~2 req/s
        "coinbase" => 300,   // Coinbase public: ~3 req/s
        _          => 500,
    };
    tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
}
