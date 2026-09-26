"""20-frame 원본 영상을 VGG16/VGG19 4096D offline cache로 변환하는 스크립트다.

주요 입력은 baseline YAML과 sequence metadata이며 주요 출력은 video별 ``.npy``와
manifest CSV다. LSTM training이나 전체 결과 평가는 하지 않는다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_fusion.data.paper_feature_dataset import (  # noqa: E402
    validate_feature_array,
    write_feature_manifest,
)
from drowsiness_fusion.data.paper_frame_dataset import PaperFrameDataset  # noqa: E402
from drowsiness_fusion.models.vgg_feature_extractor import (  # noqa: E402
    VGGFrameFeatureExtractor,
)
from drowsiness_fusion.training.paper_trainer import resolve_device  # noqa: E402


def load_config(path: Path) -> dict:
    """Baseline YAML을 읽어 mapping으로 반환하며 파일/형식 오류를 즉시 알린다."""

    if not path.is_file():
        raise FileNotFoundError(f"config 파일이 없습니다: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"config 최상위 값은 mapping이어야 합니다: {path}")
    return config


def main() -> None:
    """전체 sequence를 VGG로 추론해 video 단위 cache와 manifest를 저장한다."""

    parser = argparse.ArgumentParser(description="Paper reconstruction VGG feature 추출")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    cnn = config["cnn"]
    extraction = config["feature_extraction"]
    if int(cnn["feature_dim"]) != 4096:
        raise ValueError("Paper reconstruction feature_dim은 4096이어야 합니다.")

    manifest_path = PROJECT_ROOT / config["data"]["feature_manifest"]
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"manifest가 이미 있습니다. --overwrite가 필요합니다: {manifest_path}")
    cache_dir = PROJECT_ROOT / cnn["feature_cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = PaperFrameDataset(
        PROJECT_ROOT / config["data"]["sequence_metadata"],
        sequence_length=int(config["sequence"]["length"]),
        input_size=tuple(int(value) for value in cnn["input_size"]),
        normalization=str(cnn["normalization"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(extraction["video_batch_size"]),
        shuffle=False,
        num_workers=int(extraction["num_workers"]),
    )
    device = resolve_device(str(extraction["device"]))
    extractor = VGGFrameFeatureExtractor(
        backbone=str(cnn["backbone"]),
        pretrained=bool(cnn["pretrained"]),
        feature_layer=str(cnn["feature_layer"]),
        freeze_backbone=bool(cnn["freeze_backbone"]),
    ).to(device)
    extractor.eval()

    manifest_rows = []
    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"]
            batch_size, sequence_length = frames.shape[:2]
            flat_frames = frames.reshape(-1, *frames.shape[2:]).to(device)
            flat_features = extractor(flat_frames)
            features = flat_features.reshape(batch_size, sequence_length, 4096).cpu().numpy()
            for index, video_id in enumerate(batch["video_id"]):
                feature_array = features[index].astype(np.float32, copy=False)
                validate_feature_array(feature_array, sequence_length, 4096)
                feature_path = cache_dir / f"{video_id}.npy"
                if feature_path.exists() and not args.overwrite:
                    raise FileExistsError(
                        f"feature cache가 이미 있습니다. --overwrite가 필요합니다: {feature_path}"
                    )
                np.save(feature_path, feature_array, allow_pickle=False)
                label_id = int(batch["label"][index])
                manifest_rows.append(
                    {
                        "video_id": video_id,
                        "label": "drowsy" if label_id == 1 else "not_drowsy",
                        "feature_path": feature_path.relative_to(PROJECT_ROOT).as_posix(),
                        "sequence_length": sequence_length,
                        "feature_dim": 4096,
                        "backbone": cnn["backbone"],
                    }
                )
            print(f"feature cache 진행: {len(manifest_rows)}/{len(dataset)} videos")

    write_feature_manifest(manifest_rows, manifest_path)
    print(f"feature manifest 생성: {manifest_path}")


if __name__ == "__main__":
    main()
