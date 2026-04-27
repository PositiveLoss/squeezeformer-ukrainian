#[cfg(any(feature = "cli", feature = "python"))]
mod arrow_utils;
#[cfg(feature = "audio-decode")]
mod audio;
#[cfg(any(feature = "cli", feature = "python"))]
mod cache;
#[cfg(feature = "cli")]
mod cli;
#[cfg(feature = "frontend")]
mod dsp;
#[cfg(feature = "cli")]
mod feature_cache;
#[cfg(feature = "python")]
mod feature_loader;
#[cfg(feature = "frontend")]
mod frontend;
#[cfg(feature = "cli")]
mod record_cache;
#[cfg(any(feature = "cli", feature = "frontend", feature = "python"))]
mod util;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "cli")]
pub use cli::run_cli;
#[cfg(feature = "frontend")]
pub use frontend::{
    extract_audio_features_from_samples, extract_w2v_bert_features_from_samples,
    paraformer_frontend_config, squeezeformer_frontend_config, w2v_bert_frontend_config,
    zipformer_frontend_config, AudioFrontendConfig, FeatureMatrix, W2vBertFrontendConfig,
};

#[cfg(all(test, feature = "cli"))]
mod tests;
