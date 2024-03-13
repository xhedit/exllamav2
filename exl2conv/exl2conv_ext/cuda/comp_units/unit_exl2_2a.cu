#include "kernel_select.cuh"

fp_gemm_half_q_half_kernel pick_gemm_half_q_half_kernel_2a(const int max_m, const int perm, bool r_weights, bool mul_r_weights)
{
    if ( r_weights && !mul_r_weights)
    {
        if (max_m == 1) return map_m_count_exl2_a<1,  true, false>::pick_gemm_half_q_half_kernel(perm);
        if (max_m == 2) return map_m_count_exl2_a<2,  true, false>::pick_gemm_half_q_half_kernel(perm);
        if (max_m == 3) return map_m_count_exl2_a<3,  true, false>::pick_gemm_half_q_half_kernel(perm);
        if (max_m == 4) return map_m_count_exl2_a<4,  true, false>::pick_gemm_half_q_half_kernel(perm);
    }
    return NULL;
}
