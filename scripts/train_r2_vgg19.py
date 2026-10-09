"""R2 VGG19 last-block fine-tuning의 fold 또는 전체 4-fold를 실행한다.

입력은 R2 YAML, 기존 20-frame metadata와 fold split이며 출력은 fold별 history,
metrics, checkpoint와 전체 실행 시 CV summary다. Offline feature cache를 읽거나 만들지
않고, full VGG19 fine-tuning·validation·early stopping·best checkpoint를 사용하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from time import perf_counter
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drowsiness_fusion.data.paper_feature_dataset import read_split_rows  # noqa: E402
from drowsiness_fusion.data.paper_frame_dataset import PaperFrameDataset  # noqa: E402
from drowsiness_fusion.data.r2_dataloader import loader_settings  # noqa: E402
from drowsiness_fusion.evaluation.classification_metrics import (  # noqa: E402
    summarize_fold_metrics,
)
from drowsiness_fusion.models.r2_vgg19_lstm import (  # noqa: E402
    R2VGG19LSTM,
    parameter_summary,
)
from drowsiness_fusion.training.r2_execution import (
    require_cuda, environment, atomic_json, save_checkpoint, load_checkpoint, gpu_memory,
)  # noqa: E402
from drowsiness_fusion.training.r2_trainer import (  # noqa: E402
    evaluate_r2,
    train_r2_one_epoch,
)
from drowsiness_fusion.utils.reproducibility import seed_everything  # noqa: E402


def load_config(path: Path) -> dict:
    """R2 YAML을 읽고 핵심 구조·정책이 사전 확정값과 같은지 검증한다."""

    if not path.is_file():
        raise FileNotFoundError(f"R2 config 파일이 없습니다: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"config 최상위 값은 mapping이어야 합니다: {path}")

    cnn = config["cnn"]
    training = config["training"]
    if config["experiment"]["condition_id"] != "R2":
        raise ValueError("R2 script는 condition_id=R2 config만 지원합니다.")
    if cnn["backbone"] != "vgg19" or cnn["feature_point"] != "fc1":
        raise ValueError("R2는 VGG19 fc1 feature point를 유지해야 합니다.")
    if cnn["pretrained"] is not True or int(cnn["feature_dim"]) != 4096:
        raise ValueError("R2는 ImageNet-pretrained VGG19의 4096D fc1을 사용합니다.")
    if cnn["cnn_training_policy"] != "last_block_finetune":
        raise ValueError("R2 cnn_training_policy는 last_block_finetune이어야 합니다.")
    if cnn["frozen_blocks"] != [1, 2, 3, 4] or cnn["trainable_blocks"] != [5]:
        raise ValueError("R2는 block1~4 frozen, block5 trainable 정책입니다.")
    if cnn["fc1_trainable"] is not False:
        raise ValueError("R2 fc1은 frozen이어야 합니다.")
    if int(config["sequence"]["length"]) != 20:
        raise ValueError("R2 sequence length는 20이어야 합니다.")
    if int(config["lstm"]["input_size"]) != 4096:
        raise ValueError("R2 LSTM input size는 4096이어야 합니다.")
    if int(config["lstm"]["hidden_size"]) != 512:
        raise ValueError("R2 LSTM hidden size는 512여야 합니다.")
    if int(config["lstm"]["num_layers"]) != 1:
        raise ValueError("R2 LSTM은 single layer여야 합니다.")
    classifier = config["classifier"]
    classifier_contract = (
        int(classifier["hidden_1"]),
        str(classifier["activation_1"]),
        int(classifier["hidden_2"]),
        str(classifier["activation_2"]),
        int(classifier["num_classes"]),
    )
    if classifier_contract != (512, "relu", 64, "sigmoid", 2):
        raise ValueError("R2 classifier는 512→ReLU→64→Sigmoid→2여야 합니다.")
    if training["optimizer"] != "adam" or int(training["epochs"]) != 30:
        raise ValueError("R2는 Adam과 fixed 30 epochs를 사용합니다.")
    if int(training["batch_size"]) != 16:
        raise ValueError("R2 batch size는 사전 확정한 16이어야 합니다.")
    training_contract = (
        float(training["cnn_learning_rate"]),
        float(training["temporal_learning_rate"]),
        float(training["weight_decay"]),
        int(training["seed"]),
        str(training["scheduler"]),
        str(training["validation"]),
        bool(training["early_stopping"]),
        str(training["checkpoint_selection"]),
    )
    expected_training = (0.00001, 0.0001, 0.00001, 42, "none", "none", False, "final_epoch")
    if training_contract != expected_training:
        raise ValueError("R2 optimizer/evaluation policy가 사전 확정 조건과 다릅니다.")
    return config


def import_mlflow():
    """Tracking이 활성화된 실제 실행에서만 MLflow를 불러온다."""

    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError("MLflow가 필요합니다. `python -m pip install -e .`를 실행하세요.") from error
    return mlflow


def main() -> None:
    """Config부터 fold loop, CV aggregation까지 추적 가능한 순서로 R2를 실행한다."""

    parser = argparse.ArgumentParser(description="R2 VGG19 last-block fine-tuning")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prefetch-factor", type=int)
    args = parser.parse_args()
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    if args.prefetch_factor is not None and args.prefetch_factor < 1:
        parser.error("--prefetch-factor must be positive")
    if args.resume_from is not None and args.fold is None:
        parser.error("--resume-from requires --fold")
    device = require_cuda()
    output_root = args.output_root or (PROJECT_ROOT / "outputs/paper_reconstruction/refinement" /
        ("R2_vgg19_lastblock_finetune_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
    if output_root.exists():
        raise FileExistsError("Use a NEW output directory; existing results are preserved.")

    # 1. Config, seed, device
    config = load_config(args.config)
    training = config["training"]
    for key in ("num_workers", "pin_memory", "persistent_workers", "prefetch_factor"):
        value = getattr(args, key)
        if value is not None:
            training[key] = value
    loader_settings(training)  # Validate before creating a run or output.
    seed = int(training["seed"])
    seed_everything(seed)
    print(json.dumps(environment(config, args.fold, device), ensure_ascii=False), flush=True)
    # 2. MLflow parent run
    mlflow_config = config["mlflow"]
    mlflow = import_mlflow() if bool(mlflow_config["enabled"]) else None
    if mlflow is not None:
        mlflow.set_tracking_uri(str(mlflow_config["tracking_uri"]))
        mlflow.set_experiment(str(mlflow_config["experiment_name"]))
    parent_context = (
        mlflow.start_run(run_name=str(mlflow_config["parent_run_name"]))
        if mlflow is not None
        else nullcontext()
    )

    with parent_context:
        print(f"output_root={output_root} resume_from={args.resume_from}", flush=True)
        if mlflow is not None:
            mlflow.set_tags({"resumed": str(args.resume_from is not None).lower(),
                             "resume_from": str(args.resume_from or "none")})
        if mlflow is not None:
            mlflow.set_tags(
                {
                    "condition_id": "R2",
                    "stage": "reconstruction_refinement",
                    "backbone": "vgg19",
                    "feature_point": "fc1",
                    "cnn_training_policy": "last_block_finetune",
                    "pretrained": "true",
                    "feature_dim": "4096",
                    "evaluation_protocol": "stratified_4fold_75_25",
                    "validation": "none",
                    "early_stopping": "false",
                    "checkpoint_selection": "final_epoch",
                    "result_role": "official",
                }
            )
            mlflow.log_params(
                {
                    "cnn_learning_rate": training["cnn_learning_rate"],
                    "temporal_learning_rate": training["temporal_learning_rate"],
                    "weight_decay": training["weight_decay"],
                    "batch_size": training["batch_size"],
                    "epochs": training["epochs"],
                    "seed": seed,
                }
            )
            mlflow.log_artifact(str(args.config.resolve()), artifact_path="config")

        # 3. Fold별 data→model→parameter groups→child run→epoch→outputs
        configured_folds = int(config["cross_validation"]["folds"])
        selected_folds = [args.fold] if args.fold is not None else list(range(1, 5))
        if configured_folds != 4:
            raise ValueError("R2 evaluation protocol은 4-fold여야 합니다.")
        split_root = PROJECT_ROOT / config["data"]["split_root"]
        sequence_metadata = PROJECT_ROOT / config["data"]["sequence_metadata"]
        fold_metrics = []

        for fold in selected_folds:
            seed_everything(seed)
            train_rows = read_split_rows(split_root / f"fold_{fold}_train.csv")
            test_rows = read_split_rows(split_root / f"fold_{fold}_holdout.csv")
            dataset_kwargs = {
                "sequence_metadata": sequence_metadata,
                "sequence_length": int(config["sequence"]["length"]),
                "input_size": tuple(int(value) for value in config["cnn"]["input_size"]),
                "normalization": str(config["cnn"]["normalization"]),
            }
            train_dataset = PaperFrameDataset(
                video_ids=[row["video_id"] for row in train_rows], **dataset_kwargs
            )
            test_dataset = PaperFrameDataset(
                video_ids=[row["video_id"] for row in test_rows], **dataset_kwargs
            )
            loader_kwargs = {
                "batch_size": int(training["batch_size"]),
                **loader_settings(training),
            }
            loader_generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(
                train_dataset,
                shuffle=True,
                generator=loader_generator,
                **loader_kwargs,
            )
            test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

            output_dir = output_root / f"fold_{fold}"
            output_dir.mkdir(parents=True, exist_ok=False)
            env = environment(config, fold, device)
            print(json.dumps(env, ensure_ascii=False), flush=True)
            atomic_json(output_dir / "experiment_environment.json", env)
            torch.cuda.reset_peak_memory_stats(device)

            model = R2VGG19LSTM(
                pretrained=bool(config["cnn"]["pretrained"]),
                sequence_length=int(config["sequence"]["length"]),
                lstm_hidden_size=int(config["lstm"]["hidden_size"]),
            ).to(device)
            counts = parameter_summary(model)
            print(
                f"fold={fold} parameters total={counts['total']:,} "
                f"trainable={counts['trainable']:,} frozen={counts['frozen']:,}"
            )

            # R2의 유일한 CNN update는 block5이며 fc1/block1~4는 group에서 제외한다.
            cnn_parameters = [
                parameter
                for parameter in model.vgg_block5.parameters()
                if parameter.requires_grad
            ]
            temporal_parameters = [
                parameter for parameter in model.temporal_parameters() if parameter.requires_grad
            ]
            cnn_ids = {id(parameter) for parameter in cnn_parameters}
            temporal_ids = {id(parameter) for parameter in temporal_parameters}
            if not cnn_parameters or not temporal_parameters or cnn_ids & temporal_ids:
                raise RuntimeError("R2 optimizer parameter group이 비었거나 중복되었습니다.")
            expected_ids = {
                id(parameter) for parameter in model.parameters() if parameter.requires_grad
            }
            if cnn_ids | temporal_ids != expected_ids:
                raise RuntimeError("R2 optimizer에서 trainable parameter가 누락되었습니다.")
            optimizer = Adam(
                [
                    {
                        "params": cnn_parameters,
                        "lr": float(training["cnn_learning_rate"]),
                    },
                    {
                        "params": temporal_parameters,
                        "lr": float(training["temporal_learning_rate"]),
                    },
                ],
                weight_decay=float(training["weight_decay"]),
            )
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
                            "train_size": len(train_dataset),
                            "test_size": len(test_dataset),
                            "cnn_learning_rate": training["cnn_learning_rate"],
                            "temporal_learning_rate": training["temporal_learning_rate"],
                            "batch_size": training["batch_size"],
                            "epochs": training["epochs"],
                            "trainable_parameters": counts["trainable"],
                            "frozen_parameters": counts["frozen"],
                        }
                    )

                if mlflow is not None:
                    mlflow.log_params(env)
                    mlflow.log_artifact(str(output_dir / "experiment_environment.json"))
                history = []
                start_epoch = 1
                if args.resume_from is not None:
                    start_epoch, history = load_checkpoint(
                        args.resume_from, model, optimizer, config, fold, loader_generator)
                print(f"fold={fold} resume={args.resume_from is not None} start_epoch={start_epoch}", flush=True)
                if mlflow is not None:
                    mlflow.set_tags({"resumed": str(args.resume_from is not None).lower(),
                                     "resume_from": str(args.resume_from or "none")})
                    mlflow.log_param("start_epoch", start_epoch)
                for epoch in range(start_epoch, int(training["epochs"]) + 1):
                    torch.cuda.synchronize(device)
                    epoch_start = perf_counter()
                    train_loss = train_r2_one_epoch(
                        model, train_loader, optimizer, criterion, device
                    )
                    torch.cuda.synchronize(device)
                    train_end = perf_counter()
                    test_metrics = evaluate_r2(model, test_loader, criterion, device)
                    torch.cuda.synchronize(device)
                    test_end = perf_counter()
                    timing = {"epoch_train_seconds": train_end-epoch_start,
                              "epoch_test_seconds": test_end-train_end,
                              "avg_train_batch_seconds": (train_end-epoch_start)/len(train_loader)}
                    history.append({"epoch": epoch, "train_loss": train_loss,
                                    **test_metrics, **timing})
                    save_start = perf_counter()
                    save_checkpoint(output_dir / "last_checkpoint.pt", model, optimizer,
                                    epoch, config, fold, history, loader_generator)
                    timing["checkpoint_save_seconds"] = perf_counter()-save_start
                    timing["epoch_total_seconds"] = perf_counter()-epoch_start
                    history[-1].update(timing)
                    atomic_json(output_dir / "history.json", history)
                    atomic_json(output_dir / "intermediate_metrics.json",
                                {"fold": fold, "result_role": "intermediate", **history[-1]})
                    atomic_json(output_dir / "gpu_memory.json", gpu_memory(device))
                    print(f"timing={timing} memory={gpu_memory(device)}", flush=True)
                    if mlflow is not None:
                        mlflow.log_metrics(timing, step=epoch)
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
                        f"accuracy={test_metrics['accuracy']:.4f} "
                        f"f1={test_metrics['f1']:.4f}"
                    )

                final_metrics = {"fold": fold, **history[-1]}
                fold_metrics.append(final_metrics)
                history_path = output_dir / "history.json"
                metrics_path = output_dir / "metrics.json"
                metadata_path = output_dir / "training_metadata.json"
                history_path.write_text(
                    json.dumps(history, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                metrics_path.write_text(
                    json.dumps(final_metrics, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                metadata_path.write_text(
                    json.dumps(
                        {
                            "condition_id": "R2",
                            "device": str(device),
                            "parameter_counts": counts,
                            "cnn_learning_rate": training["cnn_learning_rate"],
                            "temporal_learning_rate": training["temporal_learning_rate"],
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
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
                    for artifact in (history_path, metrics_path, metadata_path):
                        mlflow.log_artifact(str(artifact), artifact_path=f"fold_{fold}")

        # 4. 전체 4-fold일 때만 CV mean/std를 official parent result로 기록한다.
        if args.fold is None:
            metric_summary = summarize_fold_metrics(fold_metrics)
            summary = {
                "fold_metrics": fold_metrics,
                "aggregation": "mean and population std (ddof=0)",
                "summary": metric_summary,
            }
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
            print(f"R2 4-fold summary 생성: {summary_path}")
        else:
            print(f"R2 fold {args.fold} 완료: {fold_metrics[0]}")


if __name__ == "__main__":
    main()
