"""Paper LSTM의 입출력 shape와 Fig.1 parameter count를 synthetic tensor로 검증한다.

실제 feature cache, optimizer 학습, checkpoint 생성은 수행하지 않는다.
"""

import pytest
import torch

from drowsiness_fusion.models.paper_lstm import PaperLSTMBaseline, parameter_counts


def test_paper_lstm_output_shape_and_deterministic_forward() -> None:
    torch.manual_seed(42)
    model = PaperLSTMBaseline().eval()
    features = torch.zeros(2, 20, 4096)

    with torch.no_grad():
        first = model(features)
        second = model(features)

    assert first.shape == (2, 2)
    assert torch.equal(first, second)


def test_paper_lstm_rejects_wrong_input_shape() -> None:
    model = PaperLSTMBaseline()
    with pytest.raises(ValueError, match="4096"):
        model(torch.zeros(1, 20, 512))


def test_fig1_parameter_counts() -> None:
    counts = parameter_counts(PaperLSTMBaseline())

    assert counts["lstm"] == 9_439_232
    assert counts["dense_1"] == 262_656
    assert counts["dense_2"] == 32_832
    assert counts["output"] == 130
    assert counts["total"] == 9_734_850
