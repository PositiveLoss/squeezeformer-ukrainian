import math
from pathlib import Path

import torch

from squeezeformer_pytorch import (
    NGramLanguageModel,
    SentencePieceTokenizer,
    SqueezeformerCTC,
    build_squeezeformer_encoder,
    squeezeformer_variant,
    tokenizer_from_dict,
)
from squeezeformer_pytorch.asr import load_lm_scorer
from squeezeformer_pytorch.data import (
    CV22Record,
    MaxFramesBatchSampler,
    SpecAugment,
    load_cv22_corpus_texts,
    transcript_is_usable,
)
from train import OptimizerChoice, _variant_defaults, build_optimizer, build_paper_scheduler


def expected_subsampled_length(length: int) -> int:
    value = length
    for _ in range(2):
        value = math.floor(value / 2)
    return value


@torch.no_grad()
def test_variant_forward_shapes() -> None:
    batch = 2
    lengths = torch.tensor([160, 123], dtype=torch.int64)
    features = torch.randn(batch, int(lengths.max().item()), 80)

    for variant in ["xs", "sm", "m"]:
        model = build_squeezeformer_encoder(variant)
        model.eval()
        outputs, output_lengths = model(features, lengths)

        cfg = squeezeformer_variant(variant)
        assert outputs.shape[0] == batch
        assert outputs.shape[2] == cfg.d_model
        expected_lengths = torch.tensor(
            [expected_subsampled_length(160), expected_subsampled_length(123)]
        )
        assert torch.equal(output_lengths, expected_lengths)
        assert outputs.shape[1] == int(output_lengths.max().item())
        assert torch.isfinite(outputs).all()


@torch.no_grad()
def test_temporal_unet_recovers_subsampled_resolution() -> None:
    model = build_squeezeformer_encoder("sm")
    model.eval()
    lengths = torch.tensor([200], dtype=torch.int64)
    features = torch.randn(1, 200, 80)

    outputs, output_lengths = model(features, lengths)

    assert output_lengths.item() == expected_subsampled_length(200)
    assert outputs.shape[1] == output_lengths.item()


def test_variant_table_matches_paper() -> None:
    expected = {
        "xs": (16, 144, 4),
        "s": (18, 196, 4),
        "sm": (16, 256, 4),
        "m": (20, 324, 4),
        "ml": (18, 512, 8),
        "l": (22, 640, 8),
    }
    for name, values in expected.items():
        cfg = squeezeformer_variant(name)
        assert (cfg.num_layers, cfg.d_model, cfg.num_heads) == values


def test_sentencepiece_tokenizer_roundtrip(tmp_path: Path) -> None:
    tokenizer = SentencePieceTokenizer.train(
        ["привіт світе", "це короткий тест", "мовна модель"],
        model_prefix=tmp_path / "spm",
        vocab_size=24,
    )
    token_ids = tokenizer.encode("привіт світе")
    assert token_ids
    restored = tokenizer_from_dict(tokenizer.to_dict())
    assert restored.decode(token_ids)


def test_ngram_lm_prefers_seen_extension(tmp_path: Path) -> None:
    lm = NGramLanguageModel.train(
        ["abba", "abba", "abbb"],
        order=3,
        alpha=0.1,
    )
    assert lm.score_extension("abb") > lm.score_extension("abz")
    save_path = tmp_path / "shallow_fusion_lm.json"
    lm.save(save_path)
    loaded = NGramLanguageModel.load(save_path)
    assert loaded.score_text("abba") > loaded.score_text("abza")


def test_lm_scorer_factory_spec_loads_saved_model(tmp_path: Path) -> None:
    lm = NGramLanguageModel.train(["це тест", "це ще тест"], order=2, alpha=0.1)
    lm_path = tmp_path / "lm.json"
    lm.save(lm_path)
    scorer = load_lm_scorer(
        f"squeezeformer_pytorch.lm:load_saved_ngram_scorer:{lm_path}"
    )
    assert scorer is not None
    assert scorer("це") > scorer("цz")


def test_load_cv22_corpus_texts_normalizes_and_deduplicates(tmp_path: Path) -> None:
    manifest = tmp_path / "train.tsv"
    manifest.write_text(
        "path\tsentence\n"
        "a.wav\t Це   Тест \n"
        "b.wav\tце тест\n"
        "c.wav\tМовна   Модель\n",
        encoding="utf-8",
    )
    texts = load_cv22_corpus_texts(tmp_path, deduplicate=False)
    assert texts == ["це тест", "це тест", "мовна модель"]
    deduped = load_cv22_corpus_texts(tmp_path, deduplicate=True)
    assert deduped == ["це тест", "мовна модель"]


def test_transcript_filter_rejects_pathological_rows() -> None:
    assert transcript_is_usable(
        "це нормальний рядок",
        min_chars=3,
        max_chars=40,
        max_symbol_ratio=0.5,
    )
    assert not transcript_is_usable("", min_chars=1, max_chars=40, max_symbol_ratio=0.5)
    assert not transcript_is_usable("а", min_chars=3, max_chars=40, max_symbol_ratio=0.5)
    assert not transcript_is_usable("!" * 8, min_chars=1, max_chars=40, max_symbol_ratio=0.5)
    assert not transcript_is_usable("дуже" * 20, min_chars=1, max_chars=10, max_symbol_ratio=0.5)


def test_max_frames_batch_sampler_respects_frame_budget() -> None:
    records = [
        CV22Record(None, None, "a", "0", "s0", estimated_frames=20),
        CV22Record(None, None, "b", "1", "s1", estimated_frames=25),
        CV22Record(None, None, "c", "2", "s2", estimated_frames=40),
        CV22Record(None, None, "d", "3", "s3", estimated_frames=45),
    ]
    sampler = MaxFramesBatchSampler(records, max_batch_frames=90, shuffle=False)
    batches = list(iter(sampler))
    assert batches
    for batch in batches:
        max_frames = max(records[index].estimated_frames for index in batch)
        assert len(batch) * max_frames <= 90


def test_time_reduction_kernel_matches_paper() -> None:
    model = build_squeezeformer_encoder("sm")
    reduction = model.time_reduce["7"]
    assert reduction.kernel_size == 3


def test_variant_time_unet_indices_match_reference_layout() -> None:
    expected = {
        "xs": ((7,), (15,)),
        "s": ((8,), (17,)),
        "sm": ((7,), (15,)),
        "m": ((9,), (19,)),
        "ml": ((8,), (17,)),
        "l": ((10,), (21,)),
    }
    for variant, (reduce_idx, recover_idx) in expected.items():
        config = squeezeformer_variant(variant)
        assert config.time_reduce_idx == reduce_idx
        assert config.time_recover_idx == recover_idx


def test_default_block_pattern_matches_paper_layout() -> None:
    config = squeezeformer_variant("sm")
    assert config.block_pattern == ("M", "s", "C", "s")


def test_paper_defaults_use_sentencepiece_compatible_variant_defaults() -> None:
    assert _variant_defaults("sm").peak_lr == 2e-3
    assert _variant_defaults("sm").num_time_masks == 5
    assert _variant_defaults("m").peak_lr == 1.5e-3
    assert _variant_defaults("m").num_time_masks == 7
    assert _variant_defaults("l").peak_lr == 1e-3
    assert _variant_defaults("l").num_time_masks == 10


def test_paper_scheduler_warmup_hold_decay() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = build_paper_scheduler(
        optimizer,
        steps_per_epoch=2,
        warmup_epochs=1,
        hold_epochs=1,
        decay_exponent=1.0,
    )
    observed = []
    for _ in range(8):
        optimizer.step()
        scheduler.step()
        observed.append(optimizer.param_groups[0]["lr"])
    assert observed[0] == 1.0
    assert observed[1] == 1.0
    assert observed[2] == 1.0
    assert observed[3] < observed[2]
    assert observed[4] < observed[3]


def test_specaugment_preserves_shape() -> None:
    augment = SpecAugment(
        num_freq_masks=2,
        freq_mask_param=4,
        num_time_masks=2,
        time_mask_max_ratio=0.2,
    )
    features = torch.ones(20, 8)
    augmented = augment(features)
    assert augmented.shape == features.shape


def test_muon_optimizer_partition_uses_encoder_hidden_weights() -> None:
    config = squeezeformer_variant("xs")
    model = SqueezeformerCTC(encoder_config=config, vocab_size=32)
    optimizers, optimizer_names = build_optimizer(
        model=model,
        optimizer_name=OptimizerChoice.MUON,
        muon_lr=1e-3,
        adamw_lr=5e-4,
        muon_weight_decay=1e-4,
        adamw_weight_decay=5e-5,
    )
    assert optimizer_names == ["muon", "adamw_aux"]
    muon_params = optimizers[0].param_groups[0]["params"]
    adamw_params = optimizers[1].param_groups[0]["params"] + optimizers[1].param_groups[1]["params"]
    assert muon_params
    assert adamw_params
    assert all(parameter.ndim == 2 for parameter in muon_params)
