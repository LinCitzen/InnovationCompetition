import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="核对 Conv＋MaxPool 的模拟或板端输出。")
    parser.add_argument("expected", type=Path, help="NumPy 参考输出 expected.npy")
    parser.add_argument("actual", type=Path, help="模拟输出或 rknn_benchmark 的 rt_output0.npy")
    args = parser.parse_args()

    expected = np.load(args.expected, allow_pickle=False)
    actual = np.load(args.actual, allow_pickle=False)
    if expected.shape != (1, 1, 4, 4):
        parser.error(f"参考输出应为 (1, 1, 4, 4)，实际为 {expected.shape}")
    if actual.shape == (1, 4, 4, 1):
        actual = actual.transpose(0, 3, 1, 2)
    if actual.shape != expected.shape:
        parser.error(f"输出形状 {actual.shape} 与参考形状 {expected.shape} 不匹配")

    np.testing.assert_allclose(actual, expected, rtol=0, atol=0.01, equal_nan=False)
    print(actual[0, 0])
    print(f"核对通过，最大绝对误差 {np.max(np.abs(actual - expected)):.6g}")


if __name__ == "__main__":
    main()
