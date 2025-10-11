# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from setuptools import setup, find_packages
import torch
import sys
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

# build custom rasterizer

# Check if CUDA is available
# On macOS, we use CPU-only version (no CUDA support)
if torch.cuda.is_available() and sys.platform != 'darwin':
    print("Building custom rasterizer with CUDA support...")
    custom_rasterizer_module = CUDAExtension(
        "custom_rasterizer_kernel",
        [
            "lib/custom_rasterizer_kernel/rasterizer.cpp",
            "lib/custom_rasterizer_kernel/grid_neighbor.cpp",
            "lib/custom_rasterizer_kernel/rasterizer_gpu.cu",
        ],
    )
else:
    print("Building custom rasterizer with CPU-only support (no CUDA)...")
    
    import os
    torch_dir = os.path.dirname(torch.__file__)
    
    # Get default include dirs but filter out CUDA-related ones
    include_dirs = []
    torch_include = os.path.join(torch_dir, 'include')
    
    # Only include non-CUDA torch headers
    include_dirs.append(torch_include)
    include_dirs.append(os.path.join(torch_include, 'torch/csrc/api/include'))
    include_dirs.append(os.path.join(torch_include, 'TH'))
    # Explicitly DO NOT include THC (Torch CUDA)
    
    # Add PyTorch library directory to rpath
    torch_lib_dir = os.path.join(torch_dir, 'lib')
    
    custom_rasterizer_module = CppExtension(
        "custom_rasterizer_kernel",
        [
            "lib/custom_rasterizer_kernel/rasterizer.cpp",
            "lib/custom_rasterizer_kernel/grid_neighbor.cpp",
        ],
        include_dirs=include_dirs,
        library_dirs=[torch_lib_dir],
        extra_compile_args={
            'cxx': [
                '-std=c++17',
                '-DUSE_CUDA=0',
                '-D__NO_CUDA__',
                '-DAT_PER_OPERATOR_HEADERS',
                '-DCPU_ONLY',
                '-Wno-c++11-narrowing',  # Allow implicit type conversions
                '-Wno-sign-compare'       # Suppress signed/unsigned comparison warnings
            ] if sys.platform == 'darwin' else ['-std=c++17', '-DUSE_CUDA=0'],
        },
        extra_link_args=[f'-Wl,-rpath,{torch_lib_dir}'] if sys.platform == 'darwin' else [],
        undef_macros=['USE_CUDA']
    )

setup(
    packages=find_packages(),
    version="0.1",
    name="custom_rasterizer",
    include_package_data=True,
    package_dir={"": "."},
    ext_modules=[
        custom_rasterizer_module,
    ],
    cmdclass={"build_ext": BuildExtension},
)
