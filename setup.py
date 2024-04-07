from setuptools import setup, Extension
from torch.utils import cpp_extension
from torch import version as torch_version
import os

extension_name = "exl2conv_ext"
verbose = False
ext_debug = False

precompile = 'EXLLAMA_NOCOMPILE' not in os.environ

windows = (os.name == "nt")

extra_cflags = ["/Ox", "/arch:AVX2"] if windows else ["-O3", "-mavx2"]

if ext_debug:
    extra_cflags += ["-ftime-report", "-DTORCH_USE_CUDA_DSA"]

extra_cuda_cflags = ["-lineinfo", "-O3"]

if torch_version.hip:
    extra_cuda_cflags += ["-DHIPBLAS_USE_HIP_HALF"]

extra_compile_args = {
    "cxx": extra_cflags,
    "nvcc": extra_cuda_cflags,
}

setup_kwargs = {
    "ext_modules": [
        cpp_extension.CUDAExtension(
            extension_name,
            [
                "exl2conv/exl2conv_ext/ext_bindings.cpp",
                "exl2conv/exl2conv_ext/ext_cache.cpp",
                "exl2conv/exl2conv_ext/ext_gemm.cpp",
                "exl2conv/exl2conv_ext/ext_norm.cpp",
                "exl2conv/exl2conv_ext/ext_qattn.cpp",
                "exl2conv/exl2conv_ext/ext_qmatrix.cpp",
                "exl2conv/exl2conv_ext/ext_qmlp.cpp",
                "exl2conv/exl2conv_ext/ext_quant.cpp",
                "exl2conv/exl2conv_ext/ext_rope.cpp",
                "exl2conv/exl2conv_ext/ext_safetensors.cpp",
                "exl2conv/exl2conv_ext/ext_sampling.cpp",
                "exl2conv/exl2conv_ext/cuda/h_add.cu",
                "exl2conv/exl2conv_ext/cuda/h_gemm.cu",
                "exl2conv/exl2conv_ext/cuda/lora.cu",
                "exl2conv/exl2conv_ext/cuda/pack_tensor.cu",
                "exl2conv/exl2conv_ext/cuda/quantize.cu",
                "exl2conv/exl2conv_ext/cuda/q_matrix.cu",
                "exl2conv/exl2conv_ext/cuda/q_attn.cu",
                "exl2conv/exl2conv_ext/cuda/q_mlp.cu",
                "exl2conv/exl2conv_ext/cuda/q_gemm.cu",
                "exl2conv/exl2conv_ext/cuda/rms_norm.cu",
                "exl2conv/exl2conv_ext/cuda/layer_norm.cu",
                "exl2conv/exl2conv_ext/cuda/head_norm.cu",
                "exl2conv/exl2conv_ext/cuda/rope.cu",
                "exl2conv/exl2conv_ext/cuda/cache.cu",
                "exl2conv/exl2conv_ext/cuda/util.cu",
                "exl2conv/exl2conv_ext/cuda/comp_units/kernel_select.cu",
                "exl2conv/exl2conv_ext/cuda/comp_units/unit_gptq_1.cu",
                "exl2conv/exl2conv_ext/cuda/comp_units/unit_gptq_2.cu",
                "exl2conv/exl2conv_ext/cuda/comp_units/unit_gptq_3.cu",
                "exl2conv/exl2conv_ext/cuda/comp_units/unit_exl2_1a.cu",
                "exl2conv/exl2conv_ext/cuda/comp_units/unit_exl2_1b.cu",
                "exl2conv/exl2conv_ext/cuda/comp_units/unit_exl2_2a.cu",
                "exl2conv/exl2conv_ext/cuda/comp_units/unit_exl2_2b.cu",
                "exl2conv/exl2conv_ext/cuda/comp_units/unit_exl2_3a.cu",
                "exl2conv/exl2conv_ext/cuda/comp_units/unit_exl2_3b.cu",
                "exl2conv/exl2conv_ext/cpp/quantize_func.cpp",
                "exl2conv/exl2conv_ext/cpp/profiling.cpp",
                "exl2conv/exl2conv_ext/cpp/sampling.cpp",
                "exl2conv/exl2conv_ext/cpp/sampling_avx2.cpp",
                "exl2conv/exl2conv_ext/cpp/safetensors.cpp"
            ],
            extra_compile_args=extra_compile_args,
            libraries=["cublas"] if windows else [],
        )],
    "cmdclass": {"build_ext": cpp_extension.BuildExtension}
} if precompile else {}

version_py = {}
with open("exl2conv/version.py", encoding = "utf8") as fp:
    exec(fp.read(), version_py)
version = version_py["__version__"]
print("Version:", version)

# version = "0.0.5"

setup(
    name = "exl2conv",
    version = version,
    long_description = "exllamav2 + exl2 conversion code",
    long_description_content_type='text/markdown',
    packages = [
        "exl2conv",
        "exl2conv.generator",
        # "exl2conv.generator.filters",
        # "exl2conv.server",
        # "exl2conv.exl2conv_ext",
        # "exl2conv.exl2conv_ext.cpp",
        # "exl2conv.exl2conv_ext.cuda",
        # "exl2conv.exl2conv_ext.cuda.quant",
        "exl2conv.conversion",
        "exl2conv.conversion.standard_cal_data",
    ],
    url = "https://github.com/xhedit/exl2conv",
    license = "MIT",
    author = "xhedit",
    install_requires = [
        "pandas",
        "ninja",
        "fastparquet",
        "torch>=2.2.0",
        "safetensors>=0.3.2",
        "sentencepiece>=0.1.97",
        "pygments",
        "websockets",
        "regex",
        "numpy"
    ],
    include_package_data = True,
    verbose = verbose,
    **setup_kwargs,
)
