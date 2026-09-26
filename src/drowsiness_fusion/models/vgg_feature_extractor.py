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
        feature_layer: str = "fc1",
        freeze_backbone: bool = True,
        vgg_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if feature_layer not in {"fc1", "fc2"}:
            raise ValueError("feature_layer는 fc1 또는 fc2여야 합니다.")
        self.backbone_name = backbone
        self.feature_layer = feature_layer
        self.vgg = vgg_model or build_torchvision_vgg(backbone, pretrained)
        self.classifier_end = 2 if feature_layer == "fc1" else 5
        if freeze_backbone:
            for parameter in self.vgg.parameters():
                parameter.requires_grad = False

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """``[N, 3, H, W]`` frame batch를 선택한 4096D feature로 변환한다."""

        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(f"frame 입력 shape가 올바르지 않습니다: {tuple(frames.shape)}")
        features = self.vgg.features(frames)
        features = self.vgg.avgpool(features)
        features = torch.flatten(features, 1)
        for layer in list(self.vgg.classifier.children())[: self.classifier_end]:
            features = layer(features)
        if features.ndim != 2 or features.shape[1] != 4096:
            raise ValueError(f"VGG feature는 [N, 4096]이어야 합니다: {tuple(features.shape)}")
        return features
