#include <torch/extension.h>

torch::Tensor batch_ctc_viterbi_alignments_cuda(
    torch::Tensor log_probs,
    torch::Tensor input_lengths,
    torch::Tensor targets,
    torch::Tensor target_lengths,
    int64_t blank_id);

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE(x, dt) TORCH_CHECK((x).scalar_type() == (dt), #x " has unexpected dtype")

torch::Tensor batch_ctc_viterbi_alignments_cuda_entry(
    torch::Tensor log_probs,
    torch::Tensor input_lengths,
    torch::Tensor targets,
    torch::Tensor target_lengths,
    int64_t blank_id) {
  CHECK_CUDA(log_probs);
  CHECK_CUDA(input_lengths);
  CHECK_CUDA(targets);
  CHECK_CUDA(target_lengths);
  CHECK_CONTIGUOUS(log_probs);
  CHECK_CONTIGUOUS(input_lengths);
  CHECK_CONTIGUOUS(targets);
  CHECK_CONTIGUOUS(target_lengths);
  CHECK_DTYPE(log_probs, torch::kFloat32);
  CHECK_DTYPE(input_lengths, torch::kInt64);
  CHECK_DTYPE(targets, torch::kInt64);
  CHECK_DTYPE(target_lengths, torch::kInt64);

  return batch_ctc_viterbi_alignments_cuda(
      log_probs, input_lengths, targets, target_lengths, blank_id);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "batch_ctc_viterbi_alignments_cuda",
      &batch_ctc_viterbi_alignments_cuda_entry,
      "Batch CTC Viterbi alignments on CUDA");
}
