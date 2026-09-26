"""Mock VGG로 4096D feature extraction shape와 deterministic 동작을 검증한다.

Torchvision pretrained weight를 다운로드하거나 실제 VGG16/VGG19 inference를 하지 않는다.
"""

import torch
from torch import nn

from drowsiness_fusion.models.vgg_feature_extractor import VGGFrameFeatureExtractor


class MockVGG(nn.Module):
    """FC extraction 계약만 검증하기 위한 작은 VGG-compatible module이다."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Conv2d(3, 1, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(1, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
        )


def test_mocked_vgg_feature_output_is_twenty_by_4096() -> None:
    extractor = VGGFrameFeatureExtractor(
        backbone="vgg19",
        pretrained=False,
        feature_layer="fc1",
        freeze_backbone=True,
        vgg_model=MockVGG(),
    ).eval()
    frames = torch.zeros(20, 3, 16, 16)

    first = extractor(frames)
    second = extractor(frames)

    assert first.shape == (20, 4096)
    assert torch.equal(first, second)
    assert all(not parameter.requires_grad for parameter in extractor.parameters())
