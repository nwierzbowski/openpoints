from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import numpy as np

setup(
    name="pointnet2_cuda",
    ext_modules=[
        CUDAExtension(
            "pointnet2_batch_cuda",
            [
                "src/pointnet2_api.cpp",
                "src/ball_query.cpp",
                "src/ball_query_gpu.cu",
                "src/group_points.cpp",
                "src/group_points_gpu.cu",
                "src/interpolate.cpp",
                "src/interpolate_gpu.cu",
                "src/sampling.cpp",
                "src/sampling_gpu.cu",
            ],
            extra_compile_args={"cxx": ["-g", "-D_GLIBCXX_USE_CXX11_ABI=0"], "nvcc": ["-O2"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    include_dirs=[np.get_include()],
)
