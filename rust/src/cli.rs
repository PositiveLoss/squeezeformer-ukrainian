use std::env;

use anyhow::Result;
use clap::Parser;
use env_logger::Env;

use crate::feature_cache::{run_feature_cache_cli, Cli};
use crate::record_cache::{run_record_cache_cli, RecordCacheCli};

pub fn run_cli() -> Result<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .format_timestamp_secs()
        .try_init()
        .ok();

    let args: Vec<_> = env::args_os().collect();
    if args
        .get(1)
        .and_then(|value| value.to_str())
        .is_some_and(|value| value == "record-cache" || value == "build-record-cache")
    {
        let mut record_cache_args = Vec::with_capacity(args.len() - 1);
        record_cache_args.push(args[0].clone());
        record_cache_args.extend(args.iter().skip(2).cloned());
        return run_record_cache_cli(RecordCacheCli::parse_from(record_cache_args));
    }

    run_feature_cache_cli(Cli::parse_from(args))
}
