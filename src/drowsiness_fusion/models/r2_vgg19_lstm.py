"""R2 VGG19 last-block fine-tuning 모델을 정의한다.

입력은 ``[B, 20, 3, 224, 224]`` RGB frame sequence이고 출력은 ``[B, 2]``
logits다. VGG19 block1~4와 fc1은 동결하고 block5, LSTM, dense classifier만
학습한다. Offline feature cache, fc2, attention, projection은 사용하지 않는다.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import nn

from .paper_lstm import KerasCompatibleLSTM
from .vgg_feature_extractor import build_torchvision_vgg


class R2VGG19LSTM(nn.Module):
    """VGG19 block5와 paper LSTM/head를 end-to-end로 연결한다.

    ``vgg_model``은 작은 targeted test에서만 사용하는 주입 지점이다. 실제 실험은
    torchvision ImageNet-pretrained VGG19를 생성한다.
    """

    def __init__(
        self,
        pretrained: bool = True,
        sequence_length: int = 20,
        lstm_hidden_size: int = 512,
        vgg_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.feature_dim = 4096

        vgg = vgg_model or build_torchvision_vgg("vgg19", pretrained)
        feature_layers = list(vgg.features.children())
        classifier_layers = list(vgg.classifier.children())
        if len(feature_layers) != 37 or len(classifier_layers) < 2:
            raise ValueError("torchvision VGG19 features/classifier 구조가 예상과 다릅니다.")

        # torchvision VGG19 features의 명시적 block 경계:
        # block1: conv1_1..conv1_2 + pool1 (0:5)
        # block2: conv2_1..conv2_2 + pool2 (5:10)
        # block3: conv3_1..conv3_4 + pool3 (10:19)
        # block4: conv4_1..conv4_4 + pool4 (19:28)
        # block5: conv5_1..conv5_4 + pool5 (28:37)
        self.vgg_block1 = nn.Sequential(*feature_layers[0:5])
        self.vgg_block2 = nn.Sequential(*feature_layers[5:10])
        self.vgg_block3 = nn.Sequential(*feature_layers[10:19])
        self.vgg_block4 = nn.Sequential(*feature_layers[19:28])
        self.vgg_block5 = nn.Sequential(*feature_layers[28:37])
        self.avgpool = vgg.avgpool
        self.fc1 = classifier_layers[0]
        self.fc1_relu = classifier_layers[1]

        self.lstm = KerasCompatibleLSTM(4096, lstm_hidden_size)
        self.classifier_fc1 = nn.Linear(lstm_hidden_size, 512)
        self.classifier_relu = nn.ReLU()
        self.classifier_fc2 = nn.Linear(512, 64)
        self.classifier_sigmoid = nn.Sigmoid()
        self.classifier_fc3 = nn.Linear(64, 2)

        self._apply_fine_tuning_policy()

    def _apply_fine_tuning_policy(self) -> None:
        """Block1~4/fc1은 동결하고 block5와 temporal model/head만 학습 가능하게 한다."""

        for block in (
            self.vgg_block1,
            self.vgg_block2,
            self.vgg_block3,
            self.vgg_block4,
        ):
            for parameter in block.parameters():
                parameter.requires_grad = False
        for parameter in self.fc1.parameters():
            parameter.requires_grad = False
        for parameter in self.vgg_block5.parameters():
            parameter.requires_grad = True

    def temporal_parameters(self) -> Iterator[nn.Parameter]:
        """Optimizer의 temporal LR group에 넣을 LSTM과 classifier parameter를 반환한다."""

        modules = (
            self.lstm,
            self.classifier_fc1,
            self.classifier_fc2,
            self.classifier_fc3,
        )
        for module in modules:
            yield from module.parameters()

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Frame sequence를 VGG19 fc1 4096D sequence로 바꿔 binary logits를 반환한다."""

        if frames.ndim != 5:
            raise ValueError(f"R2 입력은 [B, T, C, H, W]여야 합니다: {tuple(frames.shape)}")
        batch_size, sequence_length, channels, height, width = frames.shape
        if sequence_length != self.sequence_length or channels != 3:
            raise ValueError(
                f"R2 입력은 [B, {self.sequence_length}, 3, H, W]여야 합니다: "
                f"{tuple(frames.shape)}"
            )

        # 각 frame에 같은 VGG19를 적용하기 위해 B와 T를 하나의 batch 축으로 합친다.
        x = frames.reshape(batch_size * sequence_length, channels, height, width)

        # 동결된 lower blocks의 activation graph를 보존할 필요가 없어 memory 사용을 줄인다.
        with torch.no_grad():
            x = self.vgg_block1(x)
            x = self.vgg_block2(x)
            x = self.vgg_block3(x)
            x = self.vgg_block4(x)

        x = self.vgg_block5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc1_relu(x)

        # Frame-wise 4096D representation을 원래 video sequence 축으로 복원한다.
        x = x.reshape(batch_size, sequence_length, self.feature_dim)
        x = self.lstm(x)
        x = self.classifier_fc1(x)
        x = self.classifier_relu(x)
        x = self.classifier_fc2(x)
        x = self.classifier_sigmoid(x)
        logits = self.classifier_fc3(x)
        return logits


def parameter_summary(model: R2VGG19LSTM) -> dict[str, int]:
    """전체/trainable/frozen parameter 수를 반환한다."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
