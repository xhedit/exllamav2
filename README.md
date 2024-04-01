# ExLlamaV2

ExLlamaV2 is an inference library for running local LLMs on modern consumer GPUs.
This is a fork of exl2 that includes the conversion utility. All of the code was written by turboderp.


### Method 1: Install from source

To install the current dev version, clone the repo and run the setup script:

```
git clone https://github.com/xhedit/exl2conv
cd exl2conv
pip install .
```

By default this will also compile and install the Torch C++ extension (`exl2conv_ext`) that the library relies on. 
You can skip this step by setting the `EXLLAMA_NOCOMPILE` environment variable:

```
EXLLAMA_NOCOMPILE= pip install .
```

This will install the "JIT version" of the package, i.e. it will install the Python components without building the
C++ extension in the process. Instead, the extension will be built the first time the library is used, then cached in 
`~/.cache/torch_extensions` for subsequent use.

### Method 2: Install from PyPI

A PyPI package is available as well. It can be installed with:

```
pip install exl2conv
```

The version available through PyPI is the JIT version (see above). Still working on a solution for distributing
prebuilt wheels via PyPI.


## EXL2 quantization

ExLlamaV2 supports the same 4-bit GPTQ models as V1, but also a new "EXL2" format. EXL2 is based on the same
optimization method as GPTQ and supports 2, 3, 4, 5, 6 and 8-bit quantization. The format allows for mixing quantization
levels within a model to achieve any average bitrate between 2 and 8 bits per weight.

Moreover, it's possible to apply multiple quantization levels to each linear layer, producing something akin to sparse 
quantization wherein more important weights (columns) are quantized with more bits. The same remapping trick that lets
ExLlama work efficiently with act-order models allows this mixing of formats to happen with little to no impact on
performance.

Parameter selection is done automatically by quantizing each matrix multiple times, measuring the quantization 
error (with respect to the chosen calibration data) for each of a number of possible settings, per layer. Finally, a
combination is chosen that minimizes the maximum quantization error over the entire model while meeting a target
average bitrate.

In my tests, this scheme allows Llama2 70B to run on a single 24 GB GPU with a 2048-token context, producing coherent 
and mostly stable output with 2.55 bits per weight. 13B models run at 2.65 bits within 8 GB of VRAM, although currently
none of them uses GQA which effectively limits the context size to 2048. In either case it's unlikely that the model
will fit alongside a desktop environment. For now.

[![chat_screenshot](doc/llama2_70b_chat_thumb.png)](doc/llama2_70b_chat.png)
[![chat_screenshot](doc/codellama_13b_instruct_thumb.png)](doc/codellama_13b_instruct.png)

### Conversion

A script is provided to quantize models. Converting large models can be somewhat slow, so be warned. The conversion
script and its options are explained in [detail here](doc/convert.md)

### Community

A test community is provided at https://discord.gg/NSFwVuCjRq 
Quanting service free of charge is provided at #bot test. The computation is generiously provided by the Bloke powered by Lambda labs. 

### Community

A test community is provided at https://discord.gg/NSFwVuCjRq 
Quanting service free of charge is provided at #bot test. The computation is generiously provided by the Bloke powered by Lambda labs. 

