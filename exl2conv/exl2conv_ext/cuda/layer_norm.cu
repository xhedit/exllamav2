#include "layer_norm.cuh"
#include "util.cuh"
#include "compat.cuh"

#if defined(USE_ROCM)
#define NUM_WARPS (1024 / warpSize)
#define WARP_SIZE (warpSize)
#else
#define NUM_WARPS 32
#define WARP_SIZE 32
#endif

// y = x * w / sqrt(row_mean(x * x) + epsilon)

#define BLOCK_SIZE WARP_SIZE
#define NUM_THREADS (NUM_WARPS * WARP_SIZE)

typedef void (*fp_layer_norm_kernel)
(
    const half*,
    const half*,
    const half*,
    half*,
    const float,
    const float,
    const int,
    const int
);

template <int blocks_per_warp>
__global__ void layer_norm_kernel
(
    const half* __restrict__ x,
    const half* __restrict__ w,
    const half* __restrict__ b,
    half* __restrict__ y,
    const float epsilon,
    const float r_dim,
    const int rows,
    const int dim
)
{
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x;
    const half2* x_row = (const half2*) (x + row * dim);
    half2* y_row = (half2*) (y + row * dim);
    const half2* w2 = (const half2*) w;
    const half2* b2 = (const half2*) b;

    float itemf[blocks_per_warp][2];

    // Compute sum for each block

    float sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < blocks_per_warp; i++)
    {
        int column = warp_id * WARP_SIZE + lane_id + NUM_THREADS * i;
        if (column >= dim / 2) break;

        half2 x2 = x_row[column];
        float f0 = __half2float(__low2half(x2));
        float f1 = __half2float(__high2half(x2));
        f0 = fmaxf(-65504.0f, fminf(f0, 65504.0f));
        f1 = fmaxf(-65504.0f, fminf(f1, 65504.0f));
        itemf[i][0] = f0;
        itemf[i][1] = f1;
        sum += f0;
        sum += f1;
    }

    // Shuffle to sum across lanes

    __shared__ float sums[NUM_WARPS];

    for(int offset = warpSize / 2; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);
    if (lane_id == 0) sums[warp_id] = sum;
    __syncthreads();

    // Load partial sums from across warps, shuffle again across lanes

    sum = sums[lane_id];
    for(int offset = warpSize / 2; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);

    // Compute mean

    float mean = sum * r_dim;

    // Compute square of distance to mean

    sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < blocks_per_warp; i++)
    {
        int column = warp_id * WARP_SIZE + lane_id + NUM_THREADS * i;
        if (column >= dim / 2) break;

        float f0 = itemf[i][0];
        float f1 = itemf[i][1];
        f0 -= mean;
        f1 -= mean;
        itemf[i][0] = f0;
        itemf[i][1] = f1;
        sum = fma(f0, f0, sum);
        sum = fma(f1, f1, sum);
    }

    // Shuffle to sum across lanes

    for(int offset = warpSize / 2; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);
    if (lane_id == 0) sums[warp_id] = sum;
    __syncthreads();

    // Load partial sums from across warps, shuffle again across lanes

    sum = sums[lane_id];
    for(int offset = warpSize / 2; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);

    // Get 1/sqrt(variance)

    float rsvar = rsqrtf(sum * r_dim + epsilon);

    // Normalize x, scaling by w

    #pragma unroll 4
    for (int i = 0; i < blocks_per_warp; i++)
    {
        int column = warp_id * WARP_SIZE + lane_id + NUM_THREADS * i;
        if (column >= dim / 2) return;
        half2 w2_ = w2[column];

        float x_itemf0 = itemf[i][0];
        float x_itemf1 = itemf[i][1];
        float w_itemf0 = __half2float(__low2half(w2_));
        float w_itemf1 = __half2float(__high2half(w2_));
        float n0 = x_itemf0 * w_itemf0 * rsvar;
        float n1 = x_itemf1 * w_itemf1 * rsvar;
        half2 nh = __halves2half2(__float2half_rn(n0), __float2half_rn(n1));
        if (b) nh = __hadd2(nh, b2[column]);  // Optional bias

        y_row[column] = nh;
    }
}

fp_layer_norm_kernel pick_layer_norm_kernel(const int blocks_per_warp)
{
    if (blocks_per_warp == 1) return layer_norm_kernel<1>;
    if (blocks_per_warp == 2) return layer_norm_kernel<2>;
    if (blocks_per_warp == 3) return layer_norm_kernel<3>;
    if (blocks_per_warp == 4) return layer_norm_kernel<4>;
    if (blocks_per_warp == 5) return layer_norm_kernel<5>;
    if (blocks_per_warp == 6) return layer_norm_kernel<6>;
    if (blocks_per_warp == 7) return layer_norm_kernel<7>;
    if (blocks_per_warp == 8) return layer_norm_kernel<8>;
	return NULL;
}

void layer_norm_cuda
(
    const half* x,
    const half* w,
    const half* b,
    half* y,
    const float epsilon,
    const int rows,
    const int dim
)
{
    dim3 blockDim, gridDim;
    blockDim.x = NUM_THREADS;
    blockDim.y = 1;
    gridDim.x = rows;
    gridDim.y = 1;

    float r_dim = 1.0f / (float) dim;

    int blocks_per_warp = DIVIDE(dim, NUM_THREADS);
    fp_layer_norm_kernel kernel = pick_layer_norm_kernel(blocks_per_warp);
    kernel<<<gridDim, blockDim>>>(x, w, b, y, epsilon, r_dim, rows, dim);
}
