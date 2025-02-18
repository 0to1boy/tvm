# GTA: Generating hihg-performance tensorized program with dual-task scheduling

This repo is based on [TVM v0.14.0](https://github.com/apache/tvm/tree/v0.14.0) and reuses some code from [AMOS](https://github.com/pku-liang/AMOS).

[**Install**](#install) | [**Tutorials**](#tutorials) | [**Cite**](#cite-us)


## What is GTA

GTA is a framework designed to generate high-performance tensorized programs for DLAs. Unlike existing deep learning compilers, GTA coordinate intrinsic-based mapping abstraction with rule-based program generation strategy, followed by the application of resource-constrained rules to eliminate ineffective tensor program candidates from the search space. Additionally, GTA employ a dual-task scheduling strategy to allocate tuning resources across multiple subgraphs of deep learning networks and their mapping candidates.


## Install
GTA requires the following dependencies:
* LLVM (recommended >= 15)
* CUDA (recommended version: 11.6)
* Python (recommended version:.7.16)
* Conda (recommended miniconda)
### 1. Download the source code
```sh
cd ~
git clone https://github.com/0to1boy/tvm.git
```

### 2. Prepare the conda environment
We recommend using miniconda to manage the dependencies.
[Miniconda official website](https://docs.anaconda.com/miniconda/)
#### 2.1 Create a new conda environment
```sh
conda create -n tvm-build python=3.7.16
conda activate tvm-build
```

#### 2.2 Install the dependencies
```sh
sudo apt-get install -y libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
```

```sh
conda install conda-build git llvmdev numpy pytest cython cmake bzip2 make scipy pillow 
```

```sh
pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple decorator attrs typing-extensions tornado psutil 'xgboost>=1.1.0' cloudpickle pebble ml_dtypes pytest-order pylint appdirs ninja
```
### 3. Configure and Build
```sh
mkdir build
cd build
cp ../cmake/config.cmake .
```
1. Edit build/config.cmake to customize the compilation options
  * Changeset(USE_CUDA OFF) to set(USE_CUDA ON) to enable the CUDA backend
2. TVM requires LLVM for CPU code generation
It is recommended to build with LLVM.
If you have installed llvmdev via conda, no further installation is required  Otherwise, you can download and build the approriate version from [LLVM releases](https://releases.llvm.org/download.html).
  * Simply set set(USE_LLVM ON) to have CMake search for an available LLVM version
```sh 
cmake ..- G Ninja
ninja
```

If you are not familiar with TVM, please stick to the following steps to configure config.cmake, otherwise, just jump to the cmake step. We recommend you to refer to the documents of [TVM](https://tvm.apache.org/docs/install/from_source.html) for details.

Export environment variables
```sh
export TVM_HOME=/path/to/tvm
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
```

### 4. Some make errors and solutions
1. Not found 'GLIBCXX_3.4.30'
```sh
~/anaconda3/envs/tvm-build/lib$ rm libstdc++.so
~/anaconda3/envs/tvm-build/lib$ rm libstdc++.so.6
~/anaconda3/envs/tvm-build/lib$ ln -s /usr/lib/x86_64-gnu/libstdc++.so.6.0.30 libstdc++.so
~/anaconda3/envs/tvm-build/lib$ ln -s /usr/lib/x86_64-gnu/libstdc++.so.6.0.30 libstdc++.so.6
```
2. No module name 'torch'
```sh
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```
3. No module name 'sklearn'
```sh
conda install scikit-learn
```
## Tutorials
We have placed the experimental test cases in the benchmark folder. You can run the following commands to run the test cases.
For example, the single operator test for GTA is located under benchmakr/GTA/single_op/conv2d.
### GPU
```sh
cd benchmark/GTA/single_op/conv2d
python mapping_conv2d_GTA.py --in_dtype float16 --out_dtype float16 --begin 0 --num 1 --trials 200
```
### CPU
When running on CPU, you need to change the `target = "cuda"` to `target = "llvm -mcpu=skylake-avx512"` in the test file. Note that this requires the CPU to support AVX-512 instructions, otherwise, the execution will fail.
```sh
cd benchmark/GTA/single_op/conv2d
python mapping_conv2d_GTA.py --in_dtype float16 --out_dtype float16 --begin 0 --num 1 --trials 200
```
Examples of running instructions for other test functions are provided within the respective test function files.

## Cite us
```
@article{xie2025gta,
  title={GTA: Generating high-performance tensorized program with dual-task scheduling},
  author={Xie, Anxing and Hu, Yonghua and Wang, Yaohua and Li, Zhe and Gao, Yuxiang and Cheng, Zenghua},
  journal={Journal of Systems Architecture},
  pages={103359},
  year={2025},
  publisher={Elsevier}
}
```