# wav2vec2-bert Audio Teacher Plan

## Goal

Add a new `audio_teacher` training path that uses a frozen `wav2vec2-bert` style acoustic model as teacher supervision for the Squeezeformer student.

This is separate from the existing `liberta` path:

- `liberta` is a text teacher over transcripts.
- `audio_teacher` is an acoustic teacher over raw waveform.

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

## Current Status

Implemented:

- CLI and checkpoint-arg plumbing for `audio_teacher`
- `FrozenAudioTeacher` runtime with `AutoProcessor` and `AutoModel`
- optional waveform return from `ASRDataset`
- waveform-aware `collate_asr_batch`
- train-loop teacher instantiation and device resolution
- student projection head in `SqueezeformerCTC`
- training loss integration
- validation loss integration
- inference export stripping `audio_teacher_projection.*` and forcing `audio_teacher=False`
- initial tests for CLI parsing and waveform batching

Not implemented yet:

- inference/evaluation runtime metadata cleanup
- full test coverage beyond the initial waveform and CLI checks

## Why This Design

This repo currently supports:

- main CTC loss
- optional AED loss
- optional LiBERTa transcript embedding distillation

`wav2vec2-bert` does not fit the current LiBERTa path because it consumes audio and returns acoustic representations, not transcript embeddings.

The safest first step is to add a separate audio teacher abstraction and start with pooled hidden-state distillation. That avoids token-space mismatch and time-axis alignment complexity.

## Implementation Plan

### 1. CLI and config plumbing

Status: done

Files:

- `squeezeformer_pytorch/training/cli.py`
- `squeezeformer_pytorch/training/runtime.py`
- `train.py`

Implemented:

- CLI flags for `audio_teacher`
- startup validation for teacher device, sample rate, weight, and local model path
- checkpoint resume resolution through `_resolve_audio_teacher_settings(...)`
- training-arg persistence through `args.*` assignment in `train.py`

### 2. Add frozen acoustic teacher runtime

Status: done

Files:

- `squeezeformer_pytorch/training/runtime.py`

Added:

- `FrozenAudioTeacher`

Responsibilities:

- load `AutoProcessor`
- load `AutoModel`
- freeze parameters
- run on requested device and dtype
- expose one method for batched waveform encoding

Current outputs:

- pooled teacher hidden state per utterance
- raw hidden states for future framewise KD

API:

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

Status: done

Files:

- `squeezeformer_pytorch/data.py`

Implemented:

- `return_waveforms: bool = False` in `ASRDataset`
- optional return of:
  - `waveform`
  - `waveform_length`
  - `sample_rate`
- mono waveform conversion for simpler teacher batching
- reuse of the same augmented waveform for student features and teacher audio when augmentation is enabled

Feature caching remains unchanged.

### 4. Extend batch collation

Status: done

Files:

- `squeezeformer_pytorch/data.py`

Implemented conditional batch fields:

- `waveforms`
- `waveform_lengths`
- `sample_rates`

Waveforms are padded only when present in all batch items.

### 5. Extend model with training-only audio teacher projection

Status: done

Files:

- `squeezeformer_pytorch/asr.py`

Planned model flags:

- `audio_teacher_enabled`
- `audio_teacher_hidden_size`
- `audio_teacher_target`

Planned training-only head:

- `audio_teacher_projection`

For v1, target the encoder output and produce one pooled student embedding per utterance.

Suggested output key from `forward(..., return_training_outputs=True)`:

- `audio_teacher_student_states`

Do not reuse the `liberta_projection` path.

### 6. Instantiate teacher in train loop

Status: done

Files:

- `train.py`

Implemented:

- `FrozenAudioTeacher` instantiation in `train.py`
- teacher device resolution similar to `liberta`
- waveform-returning dataset mode only when `audio_teacher` is active

### 7. Add training loss

Status: done

Files:

- `train.py`

After student forward pass:

- run teacher on `batch["waveforms"]`
- get pooled teacher hidden states
- compare with student projected encoder states

Loss options for v1:

- normalized `MSE`
- cosine-style loss

Recommended default:

- normalized `MSE`

Apply:

```python
loss = loss + args.audio_teacher_weight * audio_teacher_loss
```

Track separately:

- `train_audio_teacher_loss`

### 8. Mirror the loss in validation

Status: done

Files:

- `squeezeformer_pytorch/training/evaluation.py`

Add:

- `audio_teacher`
- `audio_teacher_weight`

Compute and report:

- `audio_teacher_loss`
- teacher timing

### 9. Strip teacher-only weights from inference export

Status: partially done

Files:

- `squeezeformer_pytorch/training/runtime.py`

Implemented:

- inference export removes `audio_teacher_projection.*`
- exported `training_args["audio_teacher"] = False`

Remaining:

- verify end-to-end once `audio_teacher_projection` exists

### 10. Runtime metadata compatibility

Status: pending

Files:

- `squeezeformer_pytorch/inference_runtime.py`
- `squeezeformer_pytorch/evaluation_runtime.py`

Ensure new training args deserialize safely, but inference does not require teacher components.

### 11. Tests

Status: started

Files:

- `tests/test_training_data_loading.py`
- `tests/test_squeezeformer.py`
- `tests/test_training_evaluation.py`
- `tests/test_inference.py`

Implemented so far:

- CLI flag parsing for `audio_teacher`
- waveform-returning dataset mode
- waveform-aware batch collation
- model forward returning `audio_teacher_student_states`
- evaluation shard merge covering `audio_teacher_loss`

Still needed:

- model forward returning `audio_teacher_student_states`
- training/eval logging of teacher loss
- inference export stripping teacher-only weights
- resume behavior when teacher args are present in checkpoint metadata

## Recommended Delivery Order

Completed:

1. CLI and training-arg plumbing
2. dataset waveform threading
3. collate waveform batching
4. frozen teacher runtime

Remaining:

5. checkpoint export cleanup verification
6. inference/evaluation runtime metadata cleanup
7. tests

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
