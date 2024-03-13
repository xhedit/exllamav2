
uintptr_t make_q_matrix
(
    torch::Tensor q_weight,
    torch::Tensor q_perm,
    torch::Tensor q_invperm,
    torch::Tensor q_scale,
    torch::Tensor q_scale_max,
    torch::Tensor q_groups,
    torch::Tensor q_group_map,
    torch::Tensor gptq_qzeros,
    torch::Tensor gptq_scales,
    torch::Tensor gptq_g_idx,
    torch::Tensor bias,
    torch::Tensor temp_dq
);

void free_q_matrix
(
    uintptr_t handle
);

void reconstruct
(
    uintptr_t q_handle,
    torch::Tensor output
);

void gemm_half_q_half
(
    torch::Tensor a,
    uintptr_t b,
    torch::Tensor c,
    bool force_cuda
);


