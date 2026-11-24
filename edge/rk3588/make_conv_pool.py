import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main():
    parser = argparse.ArgumentParser(description="生成 Conv＋MaxPool 模型、输入和参考结果。")
    parser.add_argument("directory", type=Path, help="模型和样例的输出目录")
    directory = parser.parse_args().directory
    directory.mkdir(parents=True, exist_ok=True)

    image = np.arange(1, 65, dtype=np.float32).reshape(1, 1, 8, 8)
    weights = np.ones((1, 1, 3, 3), dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["x", "w"], ["conv"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
            helper.make_node("MaxPool", ["conv"], ["y"], kernel_shape=[2, 2], strides=[2, 2]),
        ],
        "conv_pool",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, image.shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 4, 4])],
        [numpy_helper.from_array(weights, name="w")],
    )
    model = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, directory / "conv_pool.onnx")

    # 以 NumPy 独立计算每个 3×3 窗口的和，再做 2×2 最大池化。
    padded = np.pad(image[0, 0], 1)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    conv = windows.sum(axis=(-2, -1), dtype=np.float32)
    pooled = conv.reshape(4, 2, 4, 2).max(axis=(1, 3))
    np.save(directory / "expected.npy", pooled.reshape(1, 1, 4, 4), allow_pickle=False)

    # Rockchip 的 rknn_benchmark 以 NHWC 读取输入文件。
    image_nhwc = np.ascontiguousarray(image.transpose(0, 2, 3, 1))
    np.save(directory / "input_nhwc.npy", image_nhwc, allow_pickle=False)
    print(f"模型和输入已写入 {directory.resolve()}")
    print("NumPy 参考输出")
    print(pooled)


if __name__ == "__main__":
    main()
