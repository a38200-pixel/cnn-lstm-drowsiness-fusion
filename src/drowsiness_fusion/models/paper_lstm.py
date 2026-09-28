"""Fig.1 parameter count에 맞춘 4096D-to-512 LSTM baseline 모델이다.

주요 입력은 ``[B, 20, 4096]`` feature sequence이고 주요 출력은 ``[B, 2]`` logits다.
Softmax, loss 계산, training loop, dropout, BatchNorm은 이 파일에서 수행하거나 추가하지 않는다.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class KerasCompatibleLSTM(nn.Module):
    """Keras parameter formula와 동일한 단일-bias 1-layer LSTM이다.

    입력은 batch-first ``[B, T, input_size]``이고 마지막 hidden state를 반환한다.
    PyTorch ``nn.LSTM``은 bias가 두 벌이라 Fig.1보다 2,048개 많으므로, 논문 수치와
    정확히 맞추기 위해 input/recurrent weight와 bias 한 벌을 명시적으로 구현한다.
    """

    def __init__(self, input_size: int = 4096, hidden_size: int = 512) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = True
        gate_size = 4 * hidden_size
        self.weight_input = nn.Parameter(torch.empty(gate_size, input_size))
        self.weight_hidden = nn.Parameter(torch.empty(gate_size, hidden_size))
        self.bias = nn.Parameter(torch.empty(gate_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """재현 가능한 global seed를 따르는 균등분포로 모든 parameter를 초기화한다."""

        bound = 1.0 / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.weight_input, -bound, bound)
        nn.init.uniform_(self.weight_hidden, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Batch-first sequence를 순회해 마지막 timestep의 hidden state를 반환한다."""

        if sequence.ndim != 3 or sequence.shape[2] != self.input_size:
            raise ValueError(
                f"LSTM 입력은 [B, T, {self.input_size}]이어야 합니다: {tuple(sequence.shape)}"
            )
        batch_size = sequence.shape[0]
        hidden = sequence.new_zeros(batch_size, self.hidden_size)
        cell = sequence.new_zeros(batch_size, self.hidden_size)
        for timestep in range(sequence.shape[1]):
            gates = F.linear(sequence[:, timestep], self.weight_input, self.bias)
            gates = gates + F.linear(hidden, self.weight_hidden)
            input_gate, forget_gate, candidate, output_gate = gates.chunk(4, dim=1)
            input_gate = torch.sigmoid(input_gate)
            forget_gate = torch.sigmoid(forget_gate)
            candidate = torch.tanh(candidate)
            output_gate = torch.sigmoid(output_gate)
            cell = forget_gate * cell + input_gate * candidate
            hidden = output_gate * torch.tanh(cell)
        return hidden


class PaperLSTMBaseline(nn.Module):
    """20×4096 feature를 Fig.1 기반 binary logits로 변환한다.

    LSTM512의 마지막 hidden state에 Dense512→ReLU→Dense64→Sigmoid→Dense2를 적용한다.
    최종 Softmax는 CrossEntropyLoss와 중복되므로 모델에 넣지 않고 evaluation에서만 쓴다.
    """

    def __init__(
        self,
        input_size: int = 4096,
        hidden_size: int = 512,
        num_layers: int = 1,
        hidden_1: int = 512,
        hidden_2: int = 64,
        num_classes: int = 2,
        sequence_length: int = 20,
    ) -> None:
        super().__init__()
        if num_layers != 1:
            raise ValueError("Paper reconstruction LSTM은 num_layers=1만 지원합니다.")
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.lstm = KerasCompatibleLSTM(input_size, hidden_size)
        self.dense_1 = nn.Linear(hidden_size, hidden_1)
        self.relu = nn.ReLU()
        self.dense_2 = nn.Linear(hidden_1, hidden_2)
        self.sigmoid = nn.Sigmoid()
        self.output = nn.Linear(hidden_2, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """``[B, 20, 4096]`` feature를 받아 ``[B, 2]`` logits를 반환한다."""

        expected_tail = (self.sequence_length, self.input_size)
        if features.ndim != 3 or tuple(features.shape[1:]) != expected_tail:
            raise ValueError(
                f"모델 입력은 [B, {self.sequence_length}, {self.input_size}]이어야 합니다: "
                f"{tuple(features.shape)}"
            )
        x = self.lstm(features)
        x = self.dense_1(x)
        x = self.relu(x)
        x = self.dense_2(x)
        x = self.sigmoid(x)
        logits = self.output(x)
        return logits


def parameter_counts(model: PaperLSTMBaseline) -> dict[str, int]:
    """LSTM과 각 Dense layer의 학습 parameter 수를 분리해 반환한다."""

    count = lambda module: sum(parameter.numel() for parameter in module.parameters())
    return {
        "lstm": count(model.lstm),
        "dense_1": count(model.dense_1),
        "dense_2": count(model.dense_2),
        "output": count(model.output),
        "total": count(model),
    }
