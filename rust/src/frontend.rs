use anyhow::Result;
use sha2::{Digest, Sha256};

use crate::dsp::{compute_audio_featurizer_features, compute_w2v_bert_features};
use crate::util::{hex_prefix, py_bool, py_float};

#[derive(Debug, Clone)]
pub struct AudioFrontendConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub preemphasis: f32,
    pub normalize_signal: bool,
    pub normalize_feature: bool,
    pub normalize_per_frame: bool,
}

#[derive(Debug, Clone)]
pub struct W2vBertFrontendConfig {
    pub model_source: String,
    pub sample_rate: u32,
    pub feature_size: usize,
    pub stride: usize,
    pub feature_dim: usize,
    pub padding_value: f32,
}

#[derive(Debug, Clone)]
pub(crate) enum FrontendConfig {
    Audio(AudioFrontendConfig),
    W2vBert(W2vBertFrontendConfig),
}

#[derive(Debug)]
pub struct FeatureMatrix {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<f32>,
}

pub fn squeezeformer_frontend_config() -> AudioFrontendConfig {
    AudioFrontendConfig {
        sample_rate: 16_000,
        n_fft: 400,
        win_length: 400,
        hop_length: 160,
        n_mels: 80,
        preemphasis: 0.97,
        normalize_signal: true,
        normalize_feature: true,
        normalize_per_frame: false,
    }
}

pub fn zipformer_frontend_config() -> AudioFrontendConfig {
    AudioFrontendConfig {
        sample_rate: 16_000,
        n_fft: 400,
        win_length: 400,
        hop_length: 160,
        n_mels: 80,
        preemphasis: 0.0,
        normalize_signal: false,
        normalize_feature: false,
        normalize_per_frame: false,
    }
}

pub fn w2v_bert_frontend_config(
    model_source: Option<String>,
    sample_rate: Option<u32>,
    feature_size: Option<usize>,
    stride: Option<usize>,
    feature_dim: Option<usize>,
    padding_value: Option<f32>,
) -> W2vBertFrontendConfig {
    let feature_size = feature_size.unwrap_or(80);
    let stride = stride.unwrap_or(2).max(1);
    W2vBertFrontendConfig {
        model_source: model_source.unwrap_or_else(|| "facebook/w2v-bert-2.0".to_string()),
        sample_rate: sample_rate.unwrap_or(16_000),
        feature_size,
        stride,
        feature_dim: feature_dim.unwrap_or(feature_size * stride),
        padding_value: padding_value.unwrap_or(1.0),
    }
}

pub fn extract_audio_features_from_samples(
    waveform: &[f32],
    sample_rate: u32,
    config: &AudioFrontendConfig,
) -> Result<FeatureMatrix> {
    let mut waveform = waveform.to_vec();
    compute_audio_featurizer_features(&mut waveform, sample_rate, config)
}

pub fn extract_w2v_bert_features_from_samples(
    waveform: &[f32],
    sample_rate: u32,
    config: &W2vBertFrontendConfig,
) -> Result<FeatureMatrix> {
    let mut waveform = waveform.to_vec();
    compute_w2v_bert_features(&mut waveform, sample_rate, config)
}

pub(crate) fn compute_features(
    mut waveform: Vec<f32>,
    sample_rate: u32,
    frontend: &FrontendConfig,
) -> Result<FeatureMatrix> {
    match frontend {
        FrontendConfig::Audio(config) => {
            compute_audio_featurizer_features(&mut waveform, sample_rate, config)
        }
        FrontendConfig::W2vBert(config) => {
            compute_w2v_bert_features(&mut waveform, sample_rate, config)
        }
    }
}

impl FrontendConfig {
    pub(crate) fn feature_dim(&self) -> usize {
        match self {
            Self::Audio(config) => config.n_mels,
            Self::W2vBert(config) => config.feature_dim,
        }
    }

    pub(crate) fn sample_rate(&self) -> u32 {
        match self {
            Self::Audio(config) => config.sample_rate,
            Self::W2vBert(config) => config.sample_rate,
        }
    }

    pub(crate) fn config_repr(&self) -> String {
        match self {
            Self::Audio(config) => format!(
                "{{'featurizer': {{'sample_rate': {}, 'n_fft': {}, 'win_length': {}, 'n_mels': {}, 'backend': 'torchaudio', 'preemphasis': {}, 'normalize_signal': {}, 'normalize_feature': {}, 'normalize_per_frame': {}, 'hop_length': {}}}}}",
                config.sample_rate,
                config.n_fft,
                config.win_length,
                config.n_mels,
                py_float(config.preemphasis),
                py_bool(config.normalize_signal),
                py_bool(config.normalize_feature),
                py_bool(config.normalize_per_frame),
                config.hop_length,
            ),
            Self::W2vBert(config) => format!(
                "{{'featurizer': {{'type': 'w2v_bert', 'model_source': '{}', 'sample_rate': {}, 'feature_size': {}, 'stride': {}, 'feature_dim': {}, 'padding_value': {}}}}}",
                config.model_source.replace('\'', "\\'"),
                config.sample_rate,
                config.feature_size,
                config.stride,
                config.feature_dim,
                py_float(config.padding_value),
            ),
        }
    }

    pub(crate) fn frontend_hash(&self) -> String {
        let digest = Sha256::digest(self.config_repr().as_bytes());
        hex_prefix(&digest, 12)
    }
}
