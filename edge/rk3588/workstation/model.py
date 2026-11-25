import torch
from torch import nn


IMAGE_SIZE = 64
CLASS_NAMES = ("empty", "bottle", "cup")


class WorkstationCNN(nn.Module):
    """根据取景框内的灰度图给出空工位、瓶子、杯子的类别分数。"""

    def __init__(self):
        super().__init__()
        channels = 8
        pooled_size = IMAGE_SIZE // 2
        self.conv = nn.Conv2d(1, channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = nn.Linear(channels * pooled_size**2, len(CLASS_NAMES))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.conv(images)
        features = self.relu(features)
        features = self.pool(features)
        return self.classifier(self.flatten(features))


def main():
    torch.manual_seed(0)
    model = WorkstationCNN().eval()
    image = torch.linspace(0, 1, steps=IMAGE_SIZE**2).reshape(1, 1, IMAGE_SIZE, IMAGE_SIZE)

    with torch.inference_mode():
        scores = model(image)

    expected_shape = (1, len(CLASS_NAMES))
    if scores.shape != expected_shape or not torch.isfinite(scores).all().item():
        raise RuntimeError("前向输出应包含三个有限的类别分数")

    print(model)
    print(f"输入形状 {tuple(image.shape)}")
    print(f"输出形状 {tuple(scores.shape)}")
    print(f"类别顺序 {', '.join(CLASS_NAMES)}")
    print("本地前向验证通过")


if __name__ == "__main__":
    main()
