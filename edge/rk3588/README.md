# RK3588 NPU 性能基线

用固定的 `8×8` 单通道输入和全 `1` 的 `3×3` 卷积核验证 Conv＋MaxPool。电脑生成模型、模拟推理和核对输出。RK3588 板使用 Rockchip 官方 `rknn_benchmark` 计时。每个命令块都标明执行位置。

## 1. 在电脑上准备工具链

打开 Bash 或 Zsh。以下命令使用当前开发电脑的 Python 3.11 和 RKNN-Toolkit2 2.5.0。课程板若有配套 SDK，按其依赖说明准备环境，并让模型转换工具和板端运行库使用配套版本。

```bash
cd '/home/sasara/文档/HEU/课程相关/微型计算机系统设计/InnovationCompetition'
git switch sasara
git branch --show-current

rknn_version=2.5.0
baseline_dir="$PWD/local/rknn-baseline-$rknn_version"
sdk_dir="$PWD/local/rknn-toolkit2-$rknn_version"
python3.11 -m venv "$PWD/local/venv-rknn-$rknn_version"
source "$PWD/local/venv-rknn-$rknn_version/bin/activate"
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  'torch==2.2.0+cpu' "rknn-toolkit2==$rknn_version"
python -m pip show rknn-toolkit2
```

后续电脑端命令在这个终端执行，保留已激活的环境和目录变量。Python 依赖和实验输出都在 `local/`。

## 2. 在电脑上生成模型并核对模拟输出

```bash
python edge/rk3588/make_conv_pool.py "$baseline_dir"
python edge/rk3588/compile_rknn.py "$baseline_dir" --simulate
python edge/rk3588/check_output.py "$baseline_dir/expected.npy" "$baseline_dir/sim_output.npy"
```

第一条生成 ONNX、NHWC 格式的固定输入和 NumPy 参考结果。第二条指定 `rk3588` 并关闭 INT8 量化，生成 `conv_pool.rknn` 和 `sim_output.npy`。第三条比较输出，首行应是 `[90, 108, 126, 135]`。模拟输出用于功能验证。

## 3. 在电脑上准备板端例程和文件

首次执行 `git clone` 下载指定版本，只取测速例程、头文件和 ARM64 运行库。板卡可以通过 SSH 接收文件，替换 `board` 里的用户名和地址后再执行复制命令。

```bash
git clone --depth 1 --filter=blob:none --sparse --branch "v$rknn_version" \
  https://github.com/airockchip/rknn-toolkit2.git "$sdk_dir"
git -C "$sdk_dir" sparse-checkout set \
  rknpu2/examples/rknn_benchmark \
  rknpu2/examples/3rdparty/stb \
  rknpu2/runtime/Linux/librknn_api/include \
  rknpu2/runtime/Linux/librknn_api/aarch64

board='用户名@板卡地址'
api_dir="$sdk_dir/rknpu2/runtime/Linux/librknn_api"
examples_dir="$sdk_dir/rknpu2/examples"
ssh "$board" 'mkdir -p ~/rk3588-baseline/lib'
scp -r "$examples_dir/rknn_benchmark/src" "$examples_dir/3rdparty/stb" \
  "$api_dir/include" "${board}:rk3588-baseline/"
scp "$api_dir/aarch64/librknnrt.so" "${board}:rk3588-baseline/lib/"
scp "$baseline_dir/conv_pool.rknn" "$baseline_dir/input_nhwc.npy" "${board}:rk3588-baseline/"
ssh "$board"
```

若课程 SDK 自带同版本的 `rknn_benchmark` 和 `lib/`，可以把它们放进板上的 `~/rk3588-baseline/`，配合复制过去的模型和输入直接执行第 4 步的计时命令。[Rockchip 官方测速例程](https://github.com/airockchip/rknn-toolkit2/tree/v2.5.0/rknpu2/examples/rknn_benchmark)

## 4. 在 RK3588 上构建例程并计时

先确认 `uname -m` 输出 `aarch64`。从源码构建需要 C++ 编译器和 zlib 开发头文件。Debian 或 Ubuntu 镜像缺少依赖时，可用 `sudo apt-get install g++ zlib1g-dev` 安装。

```bash
uname -m
uname -r
cd ~/rk3588-baseline
c++ -std=c++14 -O2 -Iinclude -I. \
  src/rknn_benchmark.cpp src/cnpy/cnpy.cpp \
  -L./lib -lrknnrt -lz -o rknn_benchmark
```

每次测试用一个新结果目录。首次用 `run-01`，重测时在计时和取回结果的命令里一并改编号。`100` 是计时次数，`1` 是单核掩码，选择第一个 NPU 核。官方例程会先绑定输入并预热，再记录 `rknn_run` 的平均耗时，最后写出 `rt_output0.npy`。

```bash
cd ~/rk3588-baseline
mkdir run-01
cd run-01
LD_LIBRARY_PATH=../lib ../rknn_benchmark ../conv_pool.rknn ../input_nhwc.npy 100 1 \
  2>&1 | tee benchmark.log
test -s rt_output0.npy
exit
```

从日志记录 Runtime、驱动版本和 `Avg Time`。若加载时提示版本不兼容，用课程板的配套 SDK 版本从第 1 步重新生成模型和运行库。[官方计时代码](https://github.com/airockchip/rknn-toolkit2/blob/v2.5.0/rknpu2/examples/rknn_benchmark/src/rknn_benchmark.cpp)

## 5. 回到电脑核对真实输出

```bash
mkdir -p "$baseline_dir/run-01"
scp "${board}:rk3588-baseline/run-01/rt_output0.npy" \
  "${board}:rk3588-baseline/run-01/benchmark.log" "$baseline_dir/run-01/"
python edge/rk3588/check_output.py "$baseline_dir/expected.npy" "$baseline_dir/run-01/rt_output0.npy"
```

板端文件通过校验后，`run-01/benchmark.log` 就是这一组输入的官方基线记录。后续用同一个模型、输入和单核配置对照自研后端，并统一计时范围。
