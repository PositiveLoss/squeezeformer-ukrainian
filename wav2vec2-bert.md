# wav2vec2-bert Audio Teacher Plan

## Goal

Add a new `audio_teacher` training path that uses a frozen `wav2vec2-bert` style acoustic model as teacher supervision for the Squeezeformer student.

This is separate from the existing `liberta` path:

- `liberta` is a text teacher over transcripts.
- `audio_teacher` will be an acoustic teacher over raw waveform.

## Scope

Initial implementation target:

- frozen Hugging Face `wav2vec2-bert` teacher
- raw waveform threaded through dataset and collate path only when enabled
- hidden-state distillation first
- pooled utterance-level matching first
- no inference-time dependency on teacher
- exported inference checkpoints strip teacher-only weights

Out of scope for v1:

- frame-by-frame alignment distillation
- teacher-driven pseudo-label generation
- tokenizer-space CTC KL distillation unless vocab compatibility is proven
- caching teacher outputs

## Why This Design

This repo currently supports:

- main CTC loss
- optional intermediate CTC loss
- optional AED loss
- optional LiBERTa transcript embedding distillation

`wav2vec2-bert` does not fit the current LiBERTa path because it consumes audio and returns acoustic representations, not transcript embeddings.

The safest first step is to add a separate audio teacher abstraction and start with pooled hidden-state distillation. That avoids token-space mismatch and time-axis alignment complexity.

## Implementation Plan

### 1. CLI and config plumbing

Files:

- `squeezeformer_pytorch/training/cli.py`
- `squeezeformer_pytorch/training/runtime.py`
- `train.py`

Add new CLI flags:

- `--audio-teacher`
- `--audio-teacher-model-name`
- `--audio-teacher-model-path`
- `--audio-teacher-device`
- `--audio-teacher-weight`
- `--audio-teacher-objective`
- `--audio-teacher-target`
- `--audio-teacher-layer`
- `--audio-teacher-sample-rate`
- `--audio-teacher-max-seconds`

Add validation rules:

- teacher weight must be `> 0` when enabled
- device must resolve cleanly
- `ctc_kl` objective should be blocked unless tokenizer compatibility is explicitly supported

Add training-args checkpoint persistence and resume resolution, mirroring the existing `liberta` pattern.

### 2. Add frozen acoustic teacher runtime

Files:

- `squeezeformer_pytorch/training/runtime.py`

Add a new class:

- `FrozenAudioTeacher`

Responsibilities:

- load `AutoProcessor`
- load `AutoModel` or `AutoModelForCTC`
- freeze parameters
- run on requested device and dtype
- expose one method for batched waveform encoding

Expected outputs for v1:

- pooled teacher hidden state per utterance
- optional raw hidden states for future framewise KD

Suggested API:

```python
class FrozenAudioTeacher:
    def encode_waveforms(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ...
```

### 3. Thread raw waveform through data loading

Files:

- `squeezeformer_pytorch/data.py`

Current dataset code loads waveform only to compute features, then drops it.

Change `ASRDataset` to optionally return waveform metadata when `audio_teacher` is enabled:

- `waveform`
- `waveform_length`
- `sample_rate`

Implementation notes:

- add `return_waveforms: bool = False` to `ASRDataset`
- keep feature caching behavior unchanged
- return mono waveform to simplify teacher batching
- preserve waveform augment behavior so student features and teacher audio see the same augmented sample when augmentation is enabled

### 4. Extend batch collation

Files:

- `squeezeformer_pytorch/data.py`

Update `collate_asr_batch` to conditionally include:

- `waveforms`
- `waveform_lengths`
- `sample_rates`

Waveforms should be padded only when present in batch items.

### 5. Extend model with training-only audio teacher projection

Files:

- `squeezeformer_pytorch/asr.py`

Add model flags:

- `audio_teacher_enabled`
- `audio_teacher_hidden_size`
- `audio_teacher_target`

Add a training-only projection head for student representations:

- `audio_teacher_projection`

For v1, target the encoder output and produce one pooled student embedding per utterance.

Suggested output key from `forward(..., return_training_outputs=True)`:

- `audio_teacher_student_states`

Do not reuse the `liberta_projection` path. Keep the concerns separate.

### 6. Instantiate teacher in train loop

Files:

- `train.py`

Instantiate `FrozenAudioTeacher` similarly to how `FrozenLibertaTeacher` is built today.

Enable waveform-returning dataset mode only when `audio_teacher` is active.

### 7. Add training loss

Files:

- `train.py`

After student forward pass:

- run teacher on `batch["waveforms"]`
- get pooled teacher hidden states
- compare with student projected encoder states

Loss options for v1:

- normalized `MSE`
- cosine embedding style loss

Recommended default:

- normalized `MSE`

Apply:

```python
loss = loss + args.audio_teacher_weight * audio_teacher_loss
```

Track separately in logs:

- `train_audio_teacher_loss`

### 8. Mirror the loss in validation

Files:

- `squeezeformer_pytorch/training/evaluation.py`

Add:

- `audio_teacher`
- `audio_teacher_weight`

Compute and report:

- `audio_teacher_loss`
- teacher timing

Include in Trackio and validation report payloads.

### 9. Strip teacher-only weights from inference export

Files:

- `squeezeformer_pytorch/training/runtime.py`

Update inference checkpoint export so it removes:

- `audio_teacher_projection.*`

Also force training args for exported inference payload to disable:

- `audio_teacher`

Inference artifacts must remain teacher-free.

### 10. Runtime metadata compatibility

Files:

- `squeezeformer_pytorch/inference_runtime.py`
- `squeezeformer_pytorch/evaluation_runtime.py`

Ensure new training args deserialize safely, but inference does not require teacher components.

### 11. Tests

Files:

- `tests/test_training_data_loading.py`
- `tests/test_squeezeformer.py`
- `tests/test_training_evaluation.py`
- `tests/test_inference.py`

Add tests for:

- waveform-returning dataset mode
- waveform-aware batch collation
- model forward returning `audio_teacher_student_states`
- training/eval logging of teacher loss
- inference export stripping teacher-only weights
- resume behavior when teacher args are present in checkpoint metadata

## Recommended Delivery Order

1. CLI and training-arg plumbing
2. dataset waveform threading
3. collate waveform batching
4. frozen teacher runtime
5. model projection head
6. training loss integration
7. validation integration
8. checkpoint export cleanup
9. tests

## v1 Success Criteria

The implementation is complete when:

- training runs with `--audio-teacher`
- the batch contains raw waveform only when needed
- the frozen teacher runs on waveform and returns pooled hidden states
- the student emits projected encoder states for distillation
- training and validation log `audio_teacher_loss`
- exported inference checkpoints do not depend on teacher modules
- existing non-teacher training remains unchanged

## Follow-up Work

After v1 is stable, possible extensions:

- framewise teacher-student alignment with time interpolation
- intermediate-layer distillation
- teacher CTC logit KL distillation
- offline pseudo-label generation for extra Ukrainian audio
- caching teacher outputs for repeated experiments
