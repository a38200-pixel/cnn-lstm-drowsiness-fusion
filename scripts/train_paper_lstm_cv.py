"""Cached VGG feature로 4-fold를 학습하고 local outputs와 MLflow에 기록한다."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_fusion.data.paper_feature_dataset import PaperFeatureDataset  # noqa: E402
from drowsiness_fusion.evaluation.classification_metrics import (  # noqa: E402
    summarize_fold_metrics,
)
from drowsiness_fusion.training.paper_trainer import (  # noqa: E402
    build_model,
    build_optimizer,
    evaluate,
    resolve_device,
    train_one_epoch,
)
from drowsiness_fusion.utils.reproducibility import seed_everything  # noqa: E402


def load_config(path: Path) -> dict:
    """YAML config를 읽어 mapping으로 반환한다."""

    if not path.is_file():
        raise FileNotFoundError(f"config 파일이 없습니다: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"config 최상위 값은 mapping이어야 합니다: {path}")
    return config


def import_mlflow():
    """Tracking이 활성화된 실행에서만 MLflow dependency를 불러온다."""

    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError(
            "MLflow tracking이 활성화되어 있습니다. `python -m pip install -e .`로 "
            "dependency를 설치하세요."
        ) from error
    return mlflow


def main() -> None:
    """Config→parent run→fold loop→CV summary 순서로 R0/R1 실험을 실행한다."""

    parser = argparse.ArgumentParser(description="Paper reconstruction 4-fold 학습")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    # 1. Config, device, MLflow parent run
    config = load_config(args.config)
    training = config["training"]
    seed = int(training["seed"])
    device = resolve_device(str(training["device"]))
    mlflow_config = config.get("mlflow", {})
    tracking_enabled = bool(mlflow_config.get("enabled", False))
    mlflow = import_mlflow() if tracking_enabled else None
    if mlflow is not None:
        mlflow.set_tracking_uri(str(mlflow_config["tracking_uri"]))
        mlflow.set_experiment(str(mlflow_config["experiment_name"]))
    parent_context = (
        mlflow.start_run(run_name=str(mlflow_config["parent_run_name"]))
        if mlflow is not None
        else nullcontext()
    )

    with parent_context:
        feature_point = str(config["cnn"].get("feature_point", "fc1"))
        condition_id = str(config.get("experiment", {}).get("condition_id", "R0"))
        if mlflow is not None:
            mlflow.set_tags(
                {
                    "stage": str(config["experiment"]["stage"]),
                    "condition_id": condition_id,
                    "backbone": str(config["cnn"]["backbone"]),
                    "feature_point": feature_point,
                    "feature_dim": str(config["cnn"]["feature_dim"]),
                    "evaluation_protocol": "stratified_4fold_75_25",
                    "validation": str(training["validation"]),
                    "early_stopping": str(training["early_stopping"]).lower(),
                    "checkpoint_selection": "final_epoch",
                    "result_role": str(config["experiment"]["result_role"]),
                    "paper_condition_type": "reconstruction_assumption",
                }
            )
            mlflow.log_params(
                {
                    "condition_id": condition_id,
                    "backbone": config["cnn"]["backbone"],
                    "feature_point": feature_point,
                    "feature_dim": config["cnn"]["feature_dim"],
                    "sequence_length": config["sequence"]["length"],
                    "lstm_hidden_size": config["lstm"]["hidden_size"],
                    "freeze_backbone": config["cnn"]["freeze_backbone"],
                }
            )
            mlflow.log_artifact(str(args.config.resolve()), artifact_path="config")

        # 2. Fold 1..4: data→model→optimizer→child run→epoch→local save
        fold_metrics = []
        folds = int(config["cross_validation"]["folds"])
        split_root = PROJECT_ROOT / config["data"]["split_root"]
        for fold in range(1, folds + 1):
            seed_everything(seed)
            dataset_kwargs = {
                "manifest_path": PROJECT_ROOT / config["data"]["feature_manifest"],
                "project_root": PROJECT_ROOT,
                "sequence_length": int(config["sequence"]["length"]),
                "feature_dim": int(config["cnn"]["feature_dim"]),
                "backbone": str(config["cnn"]["backbone"]),
            }
            train_dataset = PaperFeatureDataset(
                split_path=split_root / f"fold_{fold}_train.csv", **dataset_kwargs
            )
            test_dataset = PaperFeatureDataset(
                split_path=split_root / f"fold_{fold}_holdout.csv", **dataset_kwargs
            )
            loader_kwargs = {
                "batch_size": int(training["batch_size"]),
                "num_workers": int(training["num_workers"]),
                "pin_memory": device.type == "cuda",
            }
            train_loader = DataLoader(
                train_dataset,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed),
                **loader_kwargs,
            )
            test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

            model = build_model(config).to(device)
            optimizer = build_optimizer(model, training)
            criterion = nn.CrossEntropyLoss()

            child_context = (
                mlflow.start_run(run_name=f"fold_{fold}", nested=True)
                if mlflow is not None
                else nullcontext()
            )
            with child_context:
                if mlflow is not None:
                    mlflow.log_params(
                        {
                            "fold": fold,
                            "condition_id": condition_id,
                            "backbone": config["cnn"]["backbone"],
                            "feature_point": feature_point,
                            "feature_dim": config["cnn"]["feature_dim"],
                            "sequence_length": config["sequence"]["length"],
                            "lstm_hidden_size": config["lstm"]["hidden_size"],
                            "freeze_backbone": config["cnn"]["freeze_backbone"],
                            "optimizer": training["optimizer"],
                            "learning_rate": training["learning_rate"],
                            "weight_decay": training["weight_decay"],
                            "batch_size": training["batch_size"],
                            "epochs": training["epochs"],
                            "seed": seed,
                            "scheduler": training["scheduler"],
                            "validation": training["validation"],
                            "early_stopping": training["early_stopping"],
                            "train_size": len(train_dataset),
                            "test_size": len(test_dataset),
                        }
                    )

                history = []
                for epoch in range(1, int(training["epochs"]) + 1):
                    train_loss = train_one_epoch(
                        model, train_loader, optimizer, criterion, device
                    )
                    test_metrics = evaluate(model, test_loader, criterion, device)
                    history.append({"epoch": epoch, "train_loss": train_loss, **test_metrics})
                    if mlflow is not None:
                        mlflow.log_metrics(
                            {
                                "train_loss": train_loss,
                                "test_loss": float(test_metrics["loss"]),
                                "test_accuracy": float(test_metrics["accuracy"]),
                                "test_precision": float(test_metrics["precision"]),
                                "test_recall": float(test_metrics["recall"]),
                                "test_f1": float(test_metrics["f1"]),
                            },
                            step=epoch,
                        )
                    print(
                        f"fold={fold} epoch={epoch}/{training['epochs']} "
                        f"train_loss={train_loss:.6f} test_loss={test_metrics['loss']:.6f} "
                        f"accuracy={test_metrics['accuracy']:.4f} f1={test_metrics['f1']:.4f}"
                    )

                final_metrics = {"fold": fold, **history[-1]}
                fold_metrics.append(final_metrics)
                output_dir = PROJECT_ROOT / config["output"]["root"] / f"fold_{fold}"
                output_dir.mkdir(parents=True, exist_ok=True)
                history_path = output_dir / "history.json"
                metrics_path = output_dir / "metrics.json"
                history_path.write_text(
                    json.dumps(history, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                metrics_path.write_text(
                    json.dumps(final_metrics, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                torch.save(
                    {"model_state_dict": model.state_dict(), "fold": fold, "config": config},
                    output_dir / "final_model.pt",
                )
                if mlflow is not None:
                    mlflow.log_metrics(
                        {
                            "final_loss": float(final_metrics["loss"]),
                            "final_accuracy": float(final_metrics["accuracy"]),
                            "final_precision": float(final_metrics["precision"]),
                            "final_recall": float(final_metrics["recall"]),
                            "final_f1": float(final_metrics["f1"]),
                        }
                    )
                    mlflow.log_artifact(str(history_path), artifact_path=f"fold_{fold}")
                    mlflow.log_artifact(str(metrics_path), artifact_path=f"fold_{fold}")

        # 3. CV mean/std, local cv_summary, parent MLflow metrics/artifact
        metric_summary = summarize_fold_metrics(fold_metrics)
        summary = {
            "fold_metrics": fold_metrics,
            "aggregation": "mean and population std (ddof=0)",
            "summary": metric_summary,
        }
        output_root = PROJECT_ROOT / config["output"]["root"]
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path = output_root / "cv_summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if mlflow is not None:
            parent_metrics = {}
            for metric_name, values in metric_summary.items():
                parent_metrics[f"cv_{metric_name}_mean"] = float(values["mean"])
                parent_metrics[f"cv_{metric_name}_std"] = float(values["std"])
            mlflow.log_metrics(parent_metrics)
            mlflow.log_artifact(str(summary_path))

    print(f"4-fold summary 생성: {summary_path}")


if __name__ == "__main__":
    main()
