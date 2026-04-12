from __future__ import annotations

import pytest


def test_build_featurizer_from_config_uses_rust_wrappers() -> None:
    from squeezeformer_pytorch.frontend import (
        RustAudioFeaturizer,
        RustW2VBertFeatureExtractor,
        build_featurizer_from_config,
        zipformer_paper_featurizer_config,
    )

    squeezeformer = build_featurizer_from_config({})
    zipformer = build_featurizer_from_config(
        zipformer_paper_featurizer_config(),
        use_zipformer=True,
    )
    w2v_bert = build_featurizer_from_config(
        {"type": "w2v_bert", "model_source": "facebook/w2v-bert-2.0"},
        use_w2v_bert=True,
    )

    assert isinstance(squeezeformer, RustAudioFeaturizer)
    assert squeezeformer.frontend_type == "squeezeformer"
    assert isinstance(zipformer, RustAudioFeaturizer)
    assert zipformer.frontend_type == "zipformer"
    assert isinstance(w2v_bert, RustW2VBertFeatureExtractor)


def test_rust_feature_extension_extracts_numpy_features() -> None:
    pytest.importorskip("feature_cache_warmer_rust")
    import numpy as np

    from feature_cache_warmer.rust_features import (
        extract_squeezeformer,
        extract_w2v_bert,
        extract_zipformer,
    )

    waveform = np.sin(np.arange(16_000, dtype=np.float32) * 0.01).astype(np.float32)

    squeezeformer = extract_squeezeformer(waveform, 16_000)
    zipformer = extract_zipformer(waveform, 16_000)
    w2v_bert = extract_w2v_bert(waveform, 16_000)

    assert squeezeformer.dtype == np.float32
    assert squeezeformer.shape[1] == 80
    assert zipformer.dtype == np.float32
    assert zipformer.shape[1] == 80
    assert w2v_bert.dtype == np.float32
    assert w2v_bert.shape[1] == 160


def test_rust_featurizer_modules_extract_torch_features() -> None:
    pytest.importorskip("feature_cache_warmer_rust")
    import torch

    from squeezeformer_pytorch.frontend import build_featurizer_from_config

    waveform = torch.sin(torch.arange(16_000, dtype=torch.float32) * 0.01)
    squeezeformer = build_featurizer_from_config({})
    w2v_bert = build_featurizer_from_config(
        {"type": "w2v_bert", "model_source": "facebook/w2v-bert-2.0"},
        use_w2v_bert=True,
    )

    squeezeformer_features = squeezeformer(waveform, 16_000)
    w2v_bert_features = w2v_bert(waveform, 16_000)

    assert squeezeformer_features.dtype == torch.float32
    assert squeezeformer_features.shape[1] == 80
    assert w2v_bert_features.dtype == torch.float32
    assert w2v_bert_features.shape[1] == 160
