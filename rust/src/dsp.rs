use anyhow::{bail, Context, Result};
use audioadapter_buffers::direct::InterleavedSlice;
use log::debug;
use rubato::{Fft, FixedSync, Resampler};
use rustfft::num_complex::{Complex32, Complex64};
use rustfft::FftPlanner;

use crate::frontend::{AudioFrontendConfig, FeatureMatrix, W2vBertFrontendConfig};

pub(crate) fn compute_audio_featurizer_features(
    waveform: &mut Vec<f32>,
    sample_rate: u32,
    config: &AudioFrontendConfig,
) -> Result<FeatureMatrix> {
    validate_audio_config(config)?;
    if sample_rate != config.sample_rate {
        *waveform = resample_to_sample_rate(waveform, sample_rate, config.sample_rate)?;
    }
    if config.normalize_signal {
        normalize_signal(waveform);
    }
    if config.preemphasis > 0.0 {
        apply_waveform_preemphasis(waveform, config.preemphasis);
    }
    let required = config.n_fft.max(config.win_length);
    if waveform.len() < required {
        waveform.resize(required, 0.0);
    }

    let window = padded_hann_window(config.win_length, config.n_fft, true);
    let powers = power_spectrogram(
        waveform,
        SpectrogramOptions {
            frame_length: config.n_fft,
            hop_length: config.hop_length,
            fft_length: config.n_fft,
            center: true,
            window: &window,
            remove_dc_offset: false,
            frame_preemphasis: None,
        },
    )?;
    let filters = mel_filter_bank(
        config.n_fft / 2 + 1,
        config.n_mels,
        config.sample_rate,
        0.0,
        config.sample_rate as f32 / 2.0,
        MelScale::Htk,
        false,
    );
    let mut features = log_mel_from_power(&powers, &filters, 1e-5);
    if config.normalize_feature {
        if config.normalize_per_frame {
            normalize_rows(&mut features, 1e-5);
        } else {
            normalize_columns(&mut features, 1e-5, false);
        }
    }
    Ok(features)
}

pub(crate) fn compute_w2v_bert_features(
    waveform: &mut Vec<f32>,
    sample_rate: u32,
    config: &W2vBertFrontendConfig,
) -> Result<FeatureMatrix> {
    if sample_rate != config.sample_rate {
        *waveform = resample_to_sample_rate(waveform, sample_rate, config.sample_rate)?;
    }
    for value in waveform.iter_mut() {
        *value *= 32768.0;
    }
    let mut features = seamless_m4t_log_mel_features(
        waveform,
        SeamlessM4TFbankOptions {
            sample_rate: config.sample_rate,
            frame_length: 400,
            hop_length: 160,
            fft_length: 512,
            num_mel_bins: config.feature_size,
            mel_floor: 1.192_092_955_078_125e-7,
            preemphasis: 0.97,
        },
    )?;
    normalize_columns_with_variance_epsilon(&mut features, 1e-7, true);
    pad_to_stride(&mut features, config.stride, config.padding_value);
    Ok(stack_strided_features(&features, config.stride))
}

fn validate_audio_config(config: &AudioFrontendConfig) -> Result<()> {
    if config.n_fft == 0 || config.win_length == 0 || config.hop_length == 0 {
        bail!("n_fft, win_length and hop_length must be greater than zero");
    }
    if config.win_length > config.n_fft {
        bail!(
            "win_length must be <= n_fft, got win_length={} n_fft={}",
            config.win_length,
            config.n_fft
        );
    }
    Ok(())
}

fn normalize_signal(waveform: &mut [f32]) {
    if waveform.is_empty() {
        return;
    }
    let mean = waveform.iter().copied().sum::<f32>() / waveform.len() as f32;
    let mut max_abs = 0.0f32;
    for value in waveform.iter_mut() {
        *value -= mean;
        max_abs = max_abs.max(value.abs());
    }
    let scale = max_abs.max(1e-6);
    for value in waveform.iter_mut() {
        *value /= scale;
    }
}

fn apply_waveform_preemphasis(waveform: &mut [f32], coefficient: f32) {
    if waveform.len() < 2 {
        return;
    }
    for index in (1..waveform.len()).rev() {
        waveform[index] -= coefficient * waveform[index - 1];
    }
}

pub(crate) fn resample_to_sample_rate(
    input: &[f32],
    src_rate: u32,
    dst_rate: u32,
) -> Result<Vec<f32>> {
    if input.is_empty() || src_rate == 0 || dst_rate == 0 || src_rate == dst_rate {
        return Ok(input.to_vec());
    }
    debug!(
        "resampling audio with rubato src_rate={} dst_rate={} input_samples={}",
        src_rate,
        dst_rate,
        input.len()
    );

    let mut resampler = Fft::<f32>::new(
        src_rate as usize,
        dst_rate as usize,
        1024,
        2,
        1,
        FixedSync::Both,
    )
    .with_context(|| {
        format!("failed to create Rubato resampler from {src_rate} Hz to {dst_rate} Hz")
    })?;

    let input_adapter = InterleavedSlice::new(input, 1, input.len())
        .context("failed to wrap mono input for Rubato")?;
    let output_capacity = resampler.process_all_needed_output_len(input.len());
    let mut output = vec![0.0f32; output_capacity];
    let mut output_adapter = InterleavedSlice::new_mut(&mut output, 1, output_capacity)
        .context("failed to wrap mono output for Rubato")?;
    let (_input_frames, output_frames) = resampler
        .process_all_into_buffer(&input_adapter, &mut output_adapter, input.len(), None)
        .with_context(|| format!("Rubato resampling failed from {src_rate} Hz to {dst_rate} Hz"))?;
    output.truncate(output_frames);
    debug!(
        "resampled audio with rubato src_rate={} dst_rate={} output_samples={}",
        src_rate,
        dst_rate,
        output.len()
    );
    Ok(output)
}

fn padded_hann_window(win_length: usize, frame_length: usize, periodic: bool) -> Vec<f32> {
    let source = hann_window(win_length, periodic);
    if win_length == frame_length {
        return source;
    }
    let mut padded = vec![0.0; frame_length];
    let offset = (frame_length - win_length) / 2;
    padded[offset..offset + win_length].copy_from_slice(&source);
    padded
}

fn hann_window(length: usize, periodic: bool) -> Vec<f32> {
    if length == 0 {
        return Vec::new();
    }
    if length == 1 {
        return vec![1.0];
    }
    let denominator = if periodic {
        length as f32
    } else {
        (length - 1) as f32
    };
    (0..length)
        .map(|index| 0.5 - 0.5 * ((2.0 * std::f32::consts::PI * index as f32) / denominator).cos())
        .collect()
}

struct SpectrogramOptions<'a> {
    frame_length: usize,
    hop_length: usize,
    fft_length: usize,
    center: bool,
    window: &'a [f32],
    remove_dc_offset: bool,
    frame_preemphasis: Option<f32>,
}

fn power_spectrogram(waveform: &[f32], options: SpectrogramOptions<'_>) -> Result<Vec<Vec<f32>>> {
    let SpectrogramOptions {
        frame_length,
        hop_length,
        fft_length,
        center,
        window,
        remove_dc_offset,
        frame_preemphasis,
    } = options;
    if frame_length == 0 || hop_length == 0 || fft_length < frame_length {
        bail!("invalid spectrogram dimensions");
    }
    if window.len() != frame_length {
        bail!(
            "window length must equal frame_length, got {} vs {}",
            window.len(),
            frame_length
        );
    }
    let padded = if center {
        reflect_pad(waveform, frame_length / 2)
    } else {
        waveform.to_vec()
    };
    if padded.len() < frame_length {
        return Ok(Vec::new());
    }
    let num_frames = 1 + (padded.len() - frame_length) / hop_length;
    let num_bins = fft_length / 2 + 1;
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_length);
    let mut output = Vec::with_capacity(num_frames);
    let mut frame = vec![0.0f32; frame_length];
    let mut buffer = vec![Complex32::new(0.0, 0.0); fft_length];

    for frame_index in 0..num_frames {
        let start = frame_index * hop_length;
        frame.copy_from_slice(&padded[start..start + frame_length]);
        if remove_dc_offset {
            let mean = frame.iter().copied().sum::<f32>() / frame.len() as f32;
            for value in frame.iter_mut() {
                *value -= mean;
            }
        }
        if let Some(coefficient) = frame_preemphasis {
            for index in (1..frame.len()).rev() {
                frame[index] -= coefficient * frame[index - 1];
            }
            frame[0] *= 1.0 - coefficient;
        }
        for value in buffer.iter_mut() {
            *value = Complex32::new(0.0, 0.0);
        }
        for index in 0..frame_length {
            buffer[index].re = frame[index] * window[index];
        }
        fft.process(&mut buffer);
        let mut bins = Vec::with_capacity(num_bins);
        for value in buffer.iter().take(num_bins) {
            bins.push(value.re.mul_add(value.re, value.im * value.im));
        }
        output.push(bins);
    }

    Ok(output)
}

fn reflect_pad(input: &[f32], pad: usize) -> Vec<f32> {
    if pad == 0 || input.is_empty() {
        return input.to_vec();
    }
    if input.len() == 1 {
        let mut output = vec![input[0]; input.len() + pad * 2];
        output[pad] = input[0];
        return output;
    }
    let len = input.len() as isize;
    let mut output = Vec::with_capacity(input.len() + pad * 2);
    for padded_index in 0..output.capacity() {
        let source_index = padded_index as isize - pad as isize;
        output.push(input[reflect_index(source_index, len) as usize]);
    }
    output
}

fn reflect_index(mut index: isize, len: isize) -> isize {
    while index < 0 || index >= len {
        if index < 0 {
            index = -index;
        } else {
            index = 2 * len - 2 - index;
        }
    }
    index
}

#[derive(Debug, Clone, Copy)]
enum MelScale {
    Htk,
    Kaldi,
}

fn mel_filter_bank(
    num_frequency_bins: usize,
    num_mel_filters: usize,
    sample_rate: u32,
    min_frequency: f32,
    max_frequency: f32,
    mel_scale: MelScale,
    triangularize_in_mel_space: bool,
) -> Vec<Vec<f32>> {
    let min_mel = hz_to_mel(min_frequency, mel_scale);
    let max_mel = hz_to_mel(max_frequency, mel_scale);
    let mel_points: Vec<f32> = (0..num_mel_filters + 2)
        .map(|index| min_mel + (max_mel - min_mel) * index as f32 / (num_mel_filters + 1) as f32)
        .collect();
    let freq_points: Vec<f32> = mel_points
        .iter()
        .map(|mel| mel_to_hz(*mel, mel_scale))
        .collect();
    let all_freqs: Vec<f32> = (0..num_frequency_bins)
        .map(|index| {
            index as f32 * (sample_rate as f32 / 2.0) / (num_frequency_bins - 1).max(1) as f32
        })
        .collect();
    let all_mels: Vec<f32> = all_freqs
        .iter()
        .map(|frequency| hz_to_mel(*frequency, mel_scale))
        .collect();
    let mut filters = vec![vec![0.0; num_frequency_bins]; num_mel_filters];

    for mel_index in 0..num_mel_filters {
        let (left, center, right, coordinates) = if triangularize_in_mel_space {
            (
                mel_points[mel_index],
                mel_points[mel_index + 1],
                mel_points[mel_index + 2],
                &all_mels,
            )
        } else {
            (
                freq_points[mel_index],
                freq_points[mel_index + 1],
                freq_points[mel_index + 2],
                &all_freqs,
            )
        };
        for (bin_index, coordinate) in coordinates.iter().enumerate() {
            let lower = (*coordinate - left) / (center - left).max(f32::MIN_POSITIVE);
            let upper = (right - *coordinate) / (right - center).max(f32::MIN_POSITIVE);
            filters[mel_index][bin_index] = lower.min(upper).max(0.0);
        }
    }
    filters
}

fn hz_to_mel(frequency: f32, scale: MelScale) -> f32 {
    match scale {
        MelScale::Htk => 2595.0 * (1.0 + frequency / 700.0).log10(),
        MelScale::Kaldi => 1127.0 * (1.0 + frequency / 700.0).ln(),
    }
}

fn hz_to_mel_f64(frequency: f64, scale: MelScale) -> f64 {
    match scale {
        MelScale::Htk => 2595.0 * (1.0 + frequency / 700.0).log10(),
        MelScale::Kaldi => 1127.0 * (1.0 + frequency / 700.0).ln(),
    }
}

fn mel_to_hz(mel: f32, scale: MelScale) -> f32 {
    match scale {
        MelScale::Htk => 700.0 * (10f32.powf(mel / 2595.0) - 1.0),
        MelScale::Kaldi => 700.0 * ((mel / 1127.0).exp() - 1.0),
    }
}

fn log_mel_from_power(powers: &[Vec<f32>], filters: &[Vec<f32>], floor: f32) -> FeatureMatrix {
    let rows = powers.len();
    let cols = filters.len();
    let mut values = Vec::with_capacity(rows * cols);
    for frame in powers {
        for filter in filters {
            let mel_energy = frame
                .iter()
                .zip(filter.iter())
                .map(|(power, weight)| power * weight)
                .sum::<f32>()
                .max(floor);
            values.push(mel_energy.ln());
        }
    }
    FeatureMatrix { rows, cols, values }
}

struct SeamlessM4TFbankOptions {
    sample_rate: u32,
    frame_length: usize,
    hop_length: usize,
    fft_length: usize,
    num_mel_bins: usize,
    mel_floor: f64,
    preemphasis: f64,
}

fn seamless_m4t_log_mel_features(
    waveform: &[f32],
    options: SeamlessM4TFbankOptions,
) -> Result<FeatureMatrix> {
    if options.frame_length == 0
        || options.hop_length == 0
        || options.fft_length < options.frame_length
    {
        bail!("invalid SeamlessM4T fbank dimensions");
    }
    if waveform.len() < options.frame_length {
        return Ok(FeatureMatrix {
            rows: 0,
            cols: options.num_mel_bins,
            values: Vec::new(),
        });
    }

    let powers = seamless_m4t_power_spectrogram(waveform, &options)?;
    let filters = seamless_m4t_mel_filter_bank(
        options.fft_length / 2 + 1,
        options.num_mel_bins,
        options.sample_rate,
        20.0,
        (options.sample_rate / 2) as f64,
    );
    let rows = powers.len();
    let cols = options.num_mel_bins;
    let mut values = Vec::with_capacity(rows * cols);
    for frame in &powers {
        for filter in &filters {
            let mel_energy = frame
                .iter()
                .zip(filter.iter())
                .map(|(power, weight)| power * weight)
                .sum::<f64>()
                .max(options.mel_floor);
            values.push(mel_energy.ln() as f32);
        }
    }
    Ok(FeatureMatrix { rows, cols, values })
}

fn seamless_m4t_power_spectrogram(
    waveform: &[f32],
    options: &SeamlessM4TFbankOptions,
) -> Result<Vec<Vec<f64>>> {
    let num_frames = 1 + (waveform.len() - options.frame_length) / options.hop_length;
    let num_bins = options.fft_length / 2 + 1;
    let window = povey_window_f64(options.frame_length);
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(options.fft_length);
    let mut output = Vec::with_capacity(num_frames);
    let mut buffer = vec![Complex64::new(0.0, 0.0); options.fft_length];

    for frame_index in 0..num_frames {
        let start = frame_index * options.hop_length;
        for value in buffer.iter_mut() {
            *value = Complex64::new(0.0, 0.0);
        }
        for index in 0..options.frame_length {
            buffer[index].re = waveform[start + index] as f64;
        }
        let mean = buffer[..options.frame_length]
            .iter()
            .map(|value| value.re)
            .sum::<f64>()
            / options.frame_length as f64;
        for value in &mut buffer[..options.frame_length] {
            value.re -= mean;
        }
        for index in (1..options.frame_length).rev() {
            buffer[index].re -= options.preemphasis * buffer[index - 1].re;
        }
        buffer[0].re *= 1.0 - options.preemphasis;
        for index in 0..options.frame_length {
            buffer[index].re *= window[index];
        }

        fft.process(&mut buffer);
        let mut bins = Vec::with_capacity(num_bins);
        for value in buffer.iter().take(num_bins) {
            // Transformers stores the FFT result in complex64 before taking power.
            let real = value.re as f32 as f64;
            let imaginary = value.im as f32 as f64;
            bins.push(real.mul_add(real, imaginary * imaginary));
        }
        output.push(bins);
    }

    Ok(output)
}

fn povey_window_f64(length: usize) -> Vec<f64> {
    if length == 0 {
        return Vec::new();
    }
    if length == 1 {
        return vec![1.0];
    }
    let denominator = (length - 1) as f64;
    (0..length)
        .map(|index| {
            let hann =
                0.5 - 0.5 * ((2.0 * std::f64::consts::PI * index as f64) / denominator).cos();
            hann.powf(0.85)
        })
        .collect()
}

fn seamless_m4t_mel_filter_bank(
    num_frequency_bins: usize,
    num_mel_filters: usize,
    sample_rate: u32,
    min_frequency: f64,
    max_frequency: f64,
) -> Vec<Vec<f64>> {
    let min_mel = hz_to_mel_f64(min_frequency, MelScale::Kaldi);
    let max_mel = hz_to_mel_f64(max_frequency, MelScale::Kaldi);
    let filter_freqs: Vec<f64> = (0..num_mel_filters + 2)
        .map(|index| min_mel + (max_mel - min_mel) * index as f64 / (num_mel_filters + 1) as f64)
        .collect();
    let fft_bin_width = sample_rate as f64 / ((num_frequency_bins - 1) * 2) as f64;
    let fft_freqs: Vec<f64> = (0..num_frequency_bins)
        .map(|index| hz_to_mel_f64(fft_bin_width * index as f64, MelScale::Kaldi))
        .collect();
    let mut filters = vec![vec![0.0; num_frequency_bins]; num_mel_filters];

    for mel_index in 0..num_mel_filters {
        let left = filter_freqs[mel_index];
        let center = filter_freqs[mel_index + 1];
        let right = filter_freqs[mel_index + 2];
        let left_width = center - left;
        let right_width = right - center;
        for (bin_index, fft_freq) in fft_freqs.iter().enumerate() {
            let down_slope = (*fft_freq - left) / left_width;
            let up_slope = (right - *fft_freq) / right_width;
            filters[mel_index][bin_index] = down_slope.min(up_slope).max(0.0);
        }
    }
    filters
}

fn normalize_columns(features: &mut FeatureMatrix, min_std: f32, unbiased: bool) {
    if features.rows == 0 || features.cols == 0 {
        return;
    }
    for col in 0..features.cols {
        let mean = (0..features.rows)
            .map(|row| features.values[row * features.cols + col])
            .sum::<f32>()
            / features.rows as f32;
        let divisor = if unbiased && features.rows > 1 {
            (features.rows - 1) as f32
        } else {
            features.rows as f32
        };
        let variance = (0..features.rows)
            .map(|row| {
                let delta = features.values[row * features.cols + col] - mean;
                delta * delta
            })
            .sum::<f32>()
            / divisor.max(1.0);
        let std = variance.sqrt().max(min_std);
        for row in 0..features.rows {
            let index = row * features.cols + col;
            features.values[index] = (features.values[index] - mean) / std;
        }
    }
}

fn normalize_columns_with_variance_epsilon(
    features: &mut FeatureMatrix,
    variance_epsilon: f32,
    unbiased: bool,
) {
    if features.rows == 0 || features.cols == 0 {
        return;
    }
    for col in 0..features.cols {
        let mean = (0..features.rows)
            .map(|row| features.values[row * features.cols + col])
            .sum::<f32>()
            / features.rows as f32;
        let divisor = if unbiased && features.rows > 1 {
            (features.rows - 1) as f32
        } else {
            features.rows as f32
        };
        let variance = (0..features.rows)
            .map(|row| {
                let delta = features.values[row * features.cols + col] - mean;
                delta * delta
            })
            .sum::<f32>()
            / divisor.max(1.0);
        let std = (variance + variance_epsilon).sqrt();
        for row in 0..features.rows {
            let index = row * features.cols + col;
            features.values[index] = (features.values[index] - mean) / std;
        }
    }
}

fn normalize_rows(features: &mut FeatureMatrix, min_std: f32) {
    if features.rows == 0 || features.cols == 0 {
        return;
    }
    for row in 0..features.rows {
        let start = row * features.cols;
        let end = start + features.cols;
        let mean = features.values[start..end].iter().copied().sum::<f32>() / features.cols as f32;
        let variance = features.values[start..end]
            .iter()
            .map(|value| {
                let delta = *value - mean;
                delta * delta
            })
            .sum::<f32>()
            / features.cols as f32;
        let std = variance.sqrt().max(min_std);
        for value in &mut features.values[start..end] {
            *value = (*value - mean) / std;
        }
    }
}

fn pad_to_stride(features: &mut FeatureMatrix, stride: usize, padding_value: f32) {
    if stride <= 1 || features.rows.is_multiple_of(stride) {
        return;
    }
    let missing = stride - (features.rows % stride);
    features
        .values
        .extend(std::iter::repeat_n(padding_value, missing * features.cols));
    features.rows += missing;
}

fn stack_strided_features(features: &FeatureMatrix, stride: usize) -> FeatureMatrix {
    if stride <= 1 {
        return FeatureMatrix {
            rows: features.rows,
            cols: features.cols,
            values: features.values.clone(),
        };
    }
    let rows = features.rows / stride;
    let cols = features.cols * stride;
    let mut values = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for stride_index in 0..stride {
            let source_row = row * stride + stride_index;
            let start = source_row * features.cols;
            values.extend_from_slice(&features.values[start..start + features.cols]);
        }
    }
    FeatureMatrix { rows, cols, values }
}
