"""VGG16/VGG19 frame에서 4096D fully-connected feature를 추출하는 모듈이다.

주요 입력은 ``[N, 3, 224, 224]`` tensor와 CNN 설정이고, 주요 출력은 ``[N, 4096]``
feature다. 영상 decode, cache 저장, LSTM 학습은 하지 않는다.
"""

from __future__ import annotations

import torch
from torch import nn


def build_torchvision_vgg(backbone: str, pretrained: bool) -> nn.Module:
    """Backbone 이름과 pretrained 설정으로 torchvision VGG를 반환한다.

    ``pretrained=True``는 ImageNet weight를 사용할 수 있게 하는 reconstruction assumption이다.
    실제 weight 다운로드 여부는 사용자 실행 환경의 torchvision cache 상태에 달려 있다.
    """

    from torchvision.models import (
        VGG16_Weights,
        VGG19_Weights,
        vgg16,
        vgg19,
    )

    if backbone == "vgg16":
        return vgg16(weights=VGG16_Weights.DEFAULT if pretrained else None)
    if backbone == "vgg19":
        return vgg19(weights=VGG19_Weights.DEFAULT if pretrained else None)
    raise ValueError(f"지원하지 않는 VGG backbone입니다: {backbone}")


class VGGFrameFeatureExtractor(nn.Module):
    """Classic VGG classifier의 선택한 4096D representation을 반환한다.

    기본 ``fc1``은 첫 Linear와 그 뒤 ReLU까지 포함한다. ``fc2``도 선택 가능하지만,
    정확한 FC 위치는 논문에 없으므로 config에서 명시한다. 반환 shape는 ``[N, 4096]``다.
    """

    def __init__(
        self,
        backbone: str,
        pretrained: bool,
        feature_point: str | None = None,
        freeze_backbone: bool = True,
        vgg_model: nn.Module | None = None,
        feature_layer: str | None = None,
    ) -> None:
        super().__init__()
        if feature_point is not None and feature_layer is not None:
            raise ValueError("feature_point와 legacy feature_layer를 동시에 지정할 수 없습니다.")
        selected_point = feature_point or feature_layer or "fc1"
        if selected_point not in {"fc1", "fc2"}:
            raise ValueError("feature_point는 fc1 또는 fc2여야 합니다.")

        self.backbone_name = backbone
        self.feature_point = selected_point
        self.feature_layer = selected_point  # 기존 호출부/기록과의 호환성

        vgg = vgg_model or build_torchvision_vgg(backbone, pretrained)
        if freeze_backbone:
            for parameter in vgg.parameters():
                parameter.requires_grad = False

        classifier = list(vgg.classifier.children())
        if len(classifier) < 5:
            raise ValueError("VGG classifier에 fc1/fc2 extraction에 필요한 layer가 없습니다.")
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.fc1 = classifier[0]
        self.relu1 = classifier[1]
        self.dropout1 = classifier[2]
        self.fc2 = classifier[3]
        self.relu2 = classifier[4]

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """``[N, 3, H, W]`` frame batch를 선택한 4096D feature로 변환한다."""

        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(f"frame 입력 shape가 올바르지 않습니다: {tuple(frames.shape)}")
        x = self.features(frames)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.relu1(x)
        if self.feature_point == "fc1":
            feature = x
        else:
            x = self.dropout1(x)
            x = self.fc2(x)
            feature = self.relu2(x)

        if feature.ndim != 2 or feature.shape[1] != 4096:
            raise ValueError(f"VGG feature는 [N, 4096]이어야 합니다: {tuple(feature.shape)}")
        return feature
