#include "kernel_select.cuh"

fp_gemm_half_q_half_kernel pick_gemm_half_q_half_kernel_3b(const int max_m, const int perm, bool r_weights, bool mul_r_weights)
{
    if ( r_weights &&  mul_r_weights)
    {
        if (max_m == 1) return map_m_count_exl2_b<1,  true,  true>::pick_gemm_half_q_half_kernel(perm);
        if (max_m == 2) return map_m_count_exl2_b<2,  true,  true>::pick_gemm_half_q_half_kernel(perm);
        if (max_m == 3) return map_m_count_exl2_b<3,  true,  true>::pick_gemm_half_q_half_kernel(perm);
        if (max_m == 4) return map_m_count_exl2_b<4,  true,  true>::pick_gemm_half_q_half_kernel(perm);
    }
    return NULL;
}
