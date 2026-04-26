#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr float NEG_INF = -1.0e30f;

__device__ __forceinline__ int64_t state_label(
    const int64_t* targets_b,
    int64_t state,
    int64_t blank_id) {
  return (state & 1LL) ? targets_b[state >> 1] : blank_id;
}

__global__ void batch_ctc_viterbi_alignments_kernel(
    const float* log_probs,
    const int64_t* input_lengths,
    const int64_t* targets,
    const int64_t* target_lengths,
    int64_t blank_id,
    int64_t max_time,
    int64_t vocab_size,
    int64_t max_target_len,
    int64_t max_states,
    int64_t* alignments,
    int64_t* back,
    float* dp_even,
    float* dp_odd) {
  const int64_t b = blockIdx.x;
  const int64_t t_len = input_lengths[b];
  const int64_t y_len = target_lengths[b];
  const int64_t states = y_len * 2 + 1;
  const int tid = threadIdx.x;

  float* even = dp_even + b * max_states;
  float* odd = dp_odd + b * max_states;
  const int64_t* targets_b = targets + b * max_target_len;
  int64_t* alignments_b = alignments + b * max_time;
  int64_t* back_b = back + b * max_time * max_states;
  const float* log_probs_b = log_probs + b * max_time * vocab_size;

  for (int64_t s = tid; s < max_states; s += blockDim.x) {
    even[s] = NEG_INF;
    odd[s] = NEG_INF;
  }
  __syncthreads();

  if (t_len <= 0 || y_len <= 0) {
    return;
  }

  if (tid == 0) {
    even[0] = log_probs_b[blank_id];
    if (states > 1) {
      even[1] = log_probs_b[targets_b[0]];
    }
  }
  __syncthreads();

  for (int64_t t = 1; t < t_len; ++t) {
    float* prev = ((t - 1) & 1LL) ? odd : even;
    float* curr = (t & 1LL) ? odd : even;

    for (int64_t s = tid; s < max_states; s += blockDim.x) {
      if (s >= states) {
        curr[s] = NEG_INF;
        continue;
      }

      const int64_t label = state_label(targets_b, s, blank_id);
      float best_score = prev[s];
      int64_t best_state = s;

      if (s > 0 && prev[s - 1] > best_score) {
        best_score = prev[s - 1];
        best_state = s - 1;
      }

      if (s > 1) {
        const int64_t prev_prev_label = state_label(targets_b, s - 2, blank_id);
        if (label != blank_id && label != prev_prev_label && prev[s - 2] > best_score) {
          best_score = prev[s - 2];
          best_state = s - 2;
        }
      }

      curr[s] = best_score + log_probs_b[t * vocab_size + label];
      back_b[t * max_states + s] = best_state;
    }
    __syncthreads();
  }

  if (tid == 0) {
    float* final_dp = ((t_len - 1) & 1LL) ? odd : even;
    int64_t final_state = states - 1;
    if (states > 1 && final_dp[states - 2] > final_dp[final_state]) {
      final_state = states - 2;
    }

    alignments_b[t_len - 1] = state_label(targets_b, final_state, blank_id);
    for (int64_t t = t_len - 1; t > 0; --t) {
      final_state = back_b[t * max_states + final_state];
      alignments_b[t - 1] = state_label(targets_b, final_state, blank_id);
    }
  }
}

}  // namespace

torch::Tensor batch_ctc_viterbi_alignments_cuda(
    torch::Tensor log_probs,
    torch::Tensor input_lengths,
    torch::Tensor targets,
    torch::Tensor target_lengths,
    int64_t blank_id) {
  const auto batch = log_probs.size(0);
  const auto max_time = log_probs.size(1);
  const auto vocab_size = log_probs.size(2);
  const auto max_target_len = targets.size(1);
  const auto max_states = max_target_len * 2 + 1;

  auto alignments = torch::full(
      {batch, max_time},
      blank_id,
      torch::TensorOptions().device(log_probs.device()).dtype(torch::kInt64));
  auto back = torch::full(
      {batch, max_time, max_states},
      -1,
      torch::TensorOptions().device(log_probs.device()).dtype(torch::kInt64));
  auto dp_even = torch::full(
      {batch, max_states},
      NEG_INF,
      torch::TensorOptions().device(log_probs.device()).dtype(torch::kFloat32));
  auto dp_odd = torch::full(
      {batch, max_states},
      NEG_INF,
      torch::TensorOptions().device(log_probs.device()).dtype(torch::kFloat32));

  constexpr int threads = 256;
  batch_ctc_viterbi_alignments_kernel<<<batch, threads>>>(
      log_probs.data_ptr<float>(),
      input_lengths.data_ptr<int64_t>(),
      targets.data_ptr<int64_t>(),
      target_lengths.data_ptr<int64_t>(),
      blank_id,
      max_time,
      vocab_size,
      max_target_len,
      max_states,
      alignments.data_ptr<int64_t>(),
      back.data_ptr<int64_t>(),
      dp_even.data_ptr<float>(),
      dp_odd.data_ptr<float>());

  AT_CUDA_CHECK(cudaGetLastError());
  return alignments;
}
