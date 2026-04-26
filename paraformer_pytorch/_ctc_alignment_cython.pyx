# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False

from libc.float cimport FLT_MAX

import numpy as np
cimport numpy as cnp


ctypedef cnp.float32_t float32_t
ctypedef cnp.int64_t int64_t


def ctc_viterbi_alignment_cython(
    cnp.ndarray[float32_t, ndim=2] log_probs,
    cnp.ndarray[int64_t, ndim=1] targets,
    long long blank_id,
):
    cdef Py_ssize_t time = log_probs.shape[0]
    cdef Py_ssize_t target_len = targets.shape[0]
    cdef Py_ssize_t states
    cdef Py_ssize_t t
    cdef Py_ssize_t s
    cdef Py_ssize_t final_state
    cdef Py_ssize_t best_state
    cdef float32_t best_score
    cdef float32_t candidate
    cdef cnp.ndarray[int64_t, ndim=1] extended
    cdef cnp.ndarray[float32_t, ndim=2] dp
    cdef cnp.ndarray[int64_t, ndim=2] back
    cdef cnp.ndarray[int64_t, ndim=1] state_path

    if target_len == 0:
        return np.full((time,), blank_id, dtype=np.int64)

    extended = np.full((target_len * 2 + 1,), blank_id, dtype=np.int64)
    extended[1::2] = targets
    states = extended.shape[0]

    dp = np.full((time, states), -FLT_MAX, dtype=np.float32)
    back = np.full((time, states), -1, dtype=np.int64)

    dp[0, 0] = log_probs[0, blank_id]
    if states > 1:
        dp[0, 1] = log_probs[0, extended[1]]

    for t in range(1, time):
        for s in range(states):
            best_score = dp[t - 1, s]
            best_state = s
            if s > 0:
                candidate = dp[t - 1, s - 1]
                if candidate > best_score:
                    best_score = candidate
                    best_state = s - 1
            if s > 1 and extended[s] != blank_id and extended[s] != extended[s - 2]:
                candidate = dp[t - 1, s - 2]
                if candidate > best_score:
                    best_score = candidate
                    best_state = s - 2
            dp[t, s] = best_score + log_probs[t, extended[s]]
            back[t, s] = best_state

    final_state = states - 1
    if states > 1 and dp[time - 1, states - 2] > dp[time - 1, final_state]:
        final_state = states - 2

    state_path = np.empty((time,), dtype=np.int64)
    state_path[time - 1] = final_state
    for t in range(time - 1, 0, -1):
        state_path[t - 1] = back[t, state_path[t]]

    return extended[state_path]


def batch_ctc_viterbi_alignments_cython(
    cnp.ndarray[float32_t, ndim=3] log_probs,
    cnp.ndarray[int64_t, ndim=1] input_lengths,
    cnp.ndarray[int64_t, ndim=2] targets,
    cnp.ndarray[int64_t, ndim=1] target_lengths,
    long long blank_id,
):
    cdef Py_ssize_t batch = log_probs.shape[0]
    cdef Py_ssize_t max_time = log_probs.shape[1]
    cdef Py_ssize_t b
    cdef Py_ssize_t t_len
    cdef Py_ssize_t y_len
    cdef cnp.ndarray[int64_t, ndim=2] alignments = np.full(
        (batch, max_time),
        blank_id,
        dtype=np.int64,
    )
    cdef cnp.ndarray[int64_t, ndim=1] alignment

    for b in range(batch):
        t_len = input_lengths[b]
        y_len = target_lengths[b]
        alignment = ctc_viterbi_alignment_cython(
            log_probs[b, :t_len, :],
            targets[b, :y_len],
            blank_id,
        )
        alignments[b, :t_len] = alignment

    return alignments
