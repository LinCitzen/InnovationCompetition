import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from rknn.api import RKNN


def require_success(status, operation):
    if status != 0:
        raise RuntimeError(f"{operation} 失败，返回码 {status}")


def main():
    parser = argparse.ArgumentParser(description="使用 RKNN-Toolkit2 编译 RK3588 基线模型。")
    parser.add_argument("directory", type=Path, help="make_conv_pool.py 生成的样例目录")
    parser.add_argument("--simulate", action="store_true", help="在本地模拟推理并保存输出")
    args = parser.parse_args()
    directory = args.directory
    output_model = directory / "conv_pool.rknn"

    with TemporaryDirectory(dir=directory) as temporary_directory:
        model = Path(temporary_directory) / output_model.name
        numpy_output = Path(temporary_directory) / "sim_output.npy"
        rknn = RKNN()
        try:
            require_success(rknn.config(target_platform="rk3588"), "rknn.config")
            require_success(rknn.load_onnx(model=str(directory / "conv_pool.onnx")), "rknn.load_onnx")
            require_success(rknn.build(do_quantization=False), "rknn.build")
            require_success(rknn.export_rknn(str(model)), "rknn.export_rknn")

            if args.simulate:
                require_success(rknn.init_runtime(), "rknn.init_runtime")
                image = np.load(directory / "input_nhwc.npy", allow_pickle=False)
                image_nchw = np.ascontiguousarray(image.transpose(0, 3, 1, 2))
                outputs = rknn.inference(inputs=[image_nchw], data_format=["nchw"])
                if outputs is None or len(outputs) != 1:
                    raise RuntimeError("模拟推理应返回一个输出张量")
                np.save(numpy_output, outputs[0], allow_pickle=False)
        finally:
            rknn.release()

        model.replace(output_model)
        if args.simulate:
            numpy_output.replace(directory / numpy_output.name)
    print(f"模型已导出到 {output_model.resolve()}")


if __name__ == "__main__":
    main()
