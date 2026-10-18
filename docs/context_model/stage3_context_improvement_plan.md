# Stage 3 — Proposed Context Model Improvement

[확실] 작성/구현 갱신일: 2026-10-18 KST. **S3-1 IMPLEMENTED / TRAINING NOT RUN**. Stage2는 COMPLETED, Stage3는 IN PROGRESS다. 아래 설계 조건을 유지해 전용 model/config/script와 targeted tests를 구현했으며 실제 학습 결과는 없다.

## 1. Motivation

[확실] Stage2는 원논문 미공개 조건의 sensitivity/refinement를 제한적으로 조사하는 단계였다. R1/R2와 R4 diagnostic으로 질문에 대한 근거를 확보했고, 비용과 추가 탐색의 해석 가능성을 고려해 **Stage2: COMPLETED**로 종료한다. 모든 미공개 조건을 검증했거나 exact reproduction을 달성했다는 뜻은 아니다.

[확실] Stage3의 목적은 reconstruction 조건을 더 맞추는 것이 아니라 **R2 representation/training policy 위에서 temporal/context architecture의 단일 변경이 추가 성능 개선을 주는가**를 검토하는 것이다. 이 목적과 이후 Behavior/Fusion은 구분한다.

## 2. Stage2 Findings

| Condition | Official F1 mean ± population std | Finding |
|---|---:|---|
| R0 frozen FC1 | 77.94% ± 1.65%p | Initial reference |
| R1 frozen FC2 | 76.91% ± 3.99%p | FC1→FC2만으로 official 개선 없음 |
| R2 block5 fine-tuning | 80.24% ± 2.22%p | R0 대비+2.30%p, R1 대비+3.33%p |
| R4 epoch15 | 80.03% ± 1.95%p | **DIAGNOSTIC ONLY**, epoch30 대비−0.22%p |

[확실] R2 변화의 중심은 R0 대비 Recall+4.21%p이고 Precision 변화는+0.17%p다. R4는 epoch15 우위1/4folds, MIXED / INCONCLUSIVE이며 새 fixed15 학습을 권장하지 않는다. Test loss 감소와 F1 개선은 일치하지 않았다.

| Stage2 candidate | Final disposition |
|---|---|
| R3 optimizer/LR | **OPTIONAL / NOT RUN** |
| R4 duration | **DIAGNOSTIC COMPLETED / MIXED / NO NEW TRAINING RECOMMENDED** |
| R5 preprocessing/sampling | **DEFER / NOT RUN** |

[확실] R3/R5가 중요하지 않다고 입증한 것은 아니다. 현재 결과에서 주요 원인으로 특정할 근거가 부족하고 추가 전면 탐색은 비용과 test 재사용 부담을 키우므로 Stage2를 더 확장하지 않는 범위 결정이다. 근거는 `../paper_reconstruction/reconstruction_refinement.md`와 `../paper_reconstruction/r4_training_duration_plan.md`다.

## 3. Reference Model

[확실] **S3-0 reference는 R2 VGG19 Block5 Fine-Tuning, epoch30 official result**로 둔다. R0/R1과 달리 선택한 CNN adaptation policy를 고정하고 temporal readout의 효과를 비교하기에 연구 흐름이 명확하다.

```text
20 RGB frames → pretrained VGG19
→ block1~4 frozen → block5 trainable
→ frozen FC1 after ReLU4096D → [B,20,4096]
→ custom single-bias LSTM512 → final hidden h20
→ Dense512/ReLU → Dense64/Sigmoid → Dense2 logits
```

| R2 official metric | Mean ± population std(ddof=0) |
|---|---:|
| Accuracy | 81.58% ± 2.19%p |
| Precision | 81.04% ± 2.89%p |
| Recall | 79.49% ± 2.06%p |
| F1 | **80.24% ± 2.22%p** |

[확실] 동일 data/folds/training/evaluation 계약을 유지하면 기존 R2 final metrics를 reference로 재사용하고 재학습하지 않는다. Fold3/4 resume과 Fold4 MLflow terminal status stale는 기록한다. 해당 Fold4 final metric/model/artifact 저장은 확인됐고 결과 미완료로 취급하지 않는다.

[확실] R2가 동일 held-out test를 관찰하며 선택된 reference라는 점을 공개한다. 이후 동일 split의 Stage3 비교는 exploratory model-development 비교이며, 완전히 독립적인 confirmatory test라고 주장하지 않는다.

## 4. Research Question

[확실] 첫 질문은 **마지막 hidden state 하나 대신 전체20개 LSTM hidden state를 동일 가중치로 읽으면, 같은 CNN/LSTM/head 조건에서 video-level F1과 Precision/Recall 균형이 달라지는가?**다.

[추측] Final-hidden readout은 마지막 state 하나에 sequence 정보를 압축한다. 전체 timestep의 state를 직접 집계하면 중간 정보를 활용할 수 있다는 가설이 있다. 현재 결과로 final-hidden이 실제 병목이거나 mean pooling이 개선된다고 확인한 것은 아니다. Mean pooling이 짧은 event를 희석할 가능성도 검토한다.

## 5. Candidate Improvements

[확실] 아래 후보는 기존 refinement 문서에 나열된 BiLSTM, Attention, Temporal Pooling, Feature Bottleneck/Projection 범위다. 효과는 미검증이며 여러 후보를 한 번에 결합하지 않는다.

| Candidate | 가설상 보완 대상 | 구조와 parameter/compute 영향 |
|---|---|---|
| BiLSTM | 양방향 temporal context | 두 방향 recurrent 계산; 방향별 hidden 폭/concat에 따라 parameter·head 입력 변경. 방향당512이면 readout1024로 늘어 공정 비교 설계 추가 필요 |
| Attention | timestep 중요도 차이 | Hidden sequence에 학습 가능한 score와 weighted readout 추가. 단순512D linear score 예시는512 weights; 더 복잡한 scorer는 별도 용량 증가 |
| **Temporal mean pooling** | final-hidden 하나에 의존하는 readout | 기존 h1~h20 평균; head 입력512 유지, **학습 parameter 증가0**, 모든 LSTM/CNN 계산 유지 |
| Bottleneck/projection | 4096D recurrent 입력의 차원·비용 | LSTM 전 projection 추가와 input 폭 변경. 예시4096→512 linear는2,097,664 parameters지만 LSTM input weight는 감소; capacity/feature 변환 효과가 함께 바뀜 |

[확실] Attention/projection 수치는 비교용 산술 예시이며 확정 구현이 아니다. BiLSTM은 hidden 폭을 줄여 head 입력512를 맞추면 방향성뿐 아니라 방향별 capacity도 달라지는 confound를 설명해야 한다.

## 6. Candidate Comparison

[확실] 다음은 설계 복잡도·비용에 대한 구조적 비교이며 성능 순위를 뜻하지 않는다.

| Candidate | 변경 규모 / 복잡도 | 해석 가능성·ablation | 비용 및 공정성 |
|---|---|---|---|
| BiLSTM | 중간~큼 | 방향성 대비 가능하나 hidden 폭/parameter 통제 필요 | Recurrent 계산 증가; offline sequence의 future context 이용 명시 필요 |
| Attention | 중간 | 학습 weighted readout과 uniform readout을 분리 가능 | Scorer parameter와 추가 RNG/초기화 관리 필요; weight가 곧 설명은 아님 |
| **Mean pooling** | **작음** | **last-hidden vs uniform-all-hidden의 직접 대비** | **새 parameter/optimizer group 없이 readout만 변경** |
| Bottleneck | 중간 | 차원 축소와 capacity 변화가 동시에 개입 | Temporal 비용 감소 가능하지만 원본-frame/CNN 비용은 유지 |

[확실] 첫 실험은 mean pooling이 구현 범위·추가 비용·보고서 설명·parameter 통제 측면에서 가장 단순하다. 연구상 더 복잡한 architecture가 반드시 더 낫다고 가정하지 않는다.

## 7. Recommended First Experiment

[확실] **S3-1: Uniform Mean Pooling over LSTM Hidden States**를 권장한다. 단일 주요 변경은 `z=h20`을 `z=(h1+...+h20)/20`으로 바꾸는 것이다. FC feature 자체를 평균내 LSTM을 없애거나 max pooling을 추가하는 실험은 아니다.

```text
R2/S3-0: [B,20,4096] → same LSTM512 → h20 [B,512] → same head
S3-1:    [B,20,4096] → same LSTM512 → [h1,...,h20]
         → arithmetic mean over time [B,512] → same head
```

[확실] 기존 single-bias gate equations, hidden512, initial states, parameter initialization과 sequence 순서를 유지한다. Padding 없이 실제20 slots를 사용하고 기존 deterministic sampling의 중복 처리 정책을 바꾸지 않는다. 별도 positional weight, attention, max pooling, projection, dropout, layer 추가를 섞지 않는다.

[확실] Custom LSTM에 optional `return_sequences=False`를 추가했다. 기본값은 기존 final hidden을 그대로 반환하고 S3-1만 True로 전체 hidden sequence를 얻어 `mean(dim=1)`을 사용한다. Gate equations/weights/state keys와 R2 모델 파일은 유지했다.

[확실] 설계상 전체132,523,778 / trainable19,174,082 / frozen113,349,696 parameters는 R2와 같다. Pooling parameter는0이며 Adam group도 block5와 LSTM/head의 기존 두 group을 유지한다. 추가 group/LR는 필요하지 않다. 실제 구현 후 parameter coverage와 counts를 검증해야 한다.

[확실] 기존 R2 fold의 학습 완료 checkpoint를 S3-1 초기 weight로 사용하지 않는다. 그러면 이미 학습한 모델에 추가30 epoch를 주는 다른 조건이 된다. S3-1은 동일 ImageNet 초기 backbone과 seed42의 temporal/head 초기화에서 새로 학습하는 계획이다. 가능하면 parameter 생성 순서를 보존해 같은 초기 trainable tensors를 확인한다.

## 8. Fixed Conditions

| Item | Fixed condition |
|---|---|
| Data / split | SUST-DDD2,074 videos, same label-stratified4-fold, drowsy positive |
| Sampling / preprocessing | same20frame indices, RGB/bilinear224×224/ImageNet normalization; no augmentation |
| CNN | ImageNet VGG19; block1~4 frozen, block5 trainable, FC1 frozen after ReLU4096D |
| Temporal/head | same single-bias LSTM512; Dense512/ReLU→Dense64/Sigmoid→Dense2 logits |
| Optimizer / weight decay | Adam / 0.00001 |
| Block5 / temporal-head LR | 0.00001 / 0.0001 |
| Batch / seed / AMP | **16 / 42 / false** |
| Duration | **fixed30**, all folds |
| Validation / scheduler / early stopping | none / none / none |
| DataLoader | workers4, pin=true, persistent=true, prefetch1 |
| Only changed factor | final-hidden readout → uniform mean-hidden readout |

[확실] R4가 MIXED였으므로15를 새 standard로 사용하지 않고 fixed30을 유지한다. Reference와 다른 duration은 첫 비교에서 제안하지 않는다. 새로운 batch/AMP/LR 조정을 결과를 보고 암묵적으로 추가하지 않으며, 필요하면 실험을 중단하고 별도 조건으로 이유를 기록한다.

## 9. Evaluation Policy

[확실] **Official은 fixed30 final epoch**의 Accuracy/Precision/Recall/F1이며 네 fold mean과 population std(ddof=0)를 보고한다. Best test epoch는 POST-HOC DIAGNOSTIC ONLY이며 no validation/no early stopping을 유지한다.

[확실] 동일 fold끼리 R2와 S3-1의 delta를 계산한다. Primary metric은 F1 mean이며 std·Recall/Precision·Accuracy·loss·fold consistency를 함께 보고한다. Pooled confusion matrix metric을 fold mean과 혼동하지 않는다. Metric이나 primary를 결과를 보고 바꾸지 않는다.

[확실] Test history 재사용, seed1개, overlapping training folds, CUDA warn_only와 resume 한계를 명시한다. Mean 차이만으로 통계적 유의성이나 실제 환경 일반화를 주장하지 않는다. 새 조합을 test score에 따라 무제한 반복하지 않는다.

## 10. Compute Considerations

[확실] S3-0는 기존 R2 결과를 사용해 추가 학습0이다. S3-1은 승인·구현 후 fixed30×4folds=120epochs의 새 학습이 필요하다. LSTM hidden states readout만 달라 CNN forward/decode와 block5 backward 비용을 없애지 못한다. R2 offline feature cache도 재사용하지 않는다.

[확실] [B=16,T=20,H=512] float32 hidden tensor 원소량은 약0.625MiB다. 이는 해당 tensor 값만의 산술량이고 autograd/stack/gradient를 포함한 실제 peak 증가가 아니다. 추가 temporal reduction 계산은 있지만 실제 overhead와 runtime 개선/악화는 NOT MEASURED다.

[추측] 기존 약70~75분/epoch를 단순 적용하면 새120epochs의 장비 점유는 약140~150시간이다. Cold-start, system RAM/commit, shared GPU memory, Windows worker 재시작과 중단에 따라 달라지는 참고값이다. GPU kernel active time과 다르며 완료 시간 보장은 아니다. 이번에는 측정이나 학습을 하지 않는다.

## 11. Decision Rule

[확실] 임의 최소 개선폭이나 유의성 threshold를 만들지 않는다. 다음 순서로 동일 결과를 평가한다.

1. [확실] 먼저 네 fold 완주, fixed conditions와 parameter counts, freeze/gradient policy, 결과 파일/MLflow 저장을 검증한다. 조건이 바뀌거나 일부 fold만 있으면 official4-fold 비교를 완성했다고 주장하지 않는다.
2. [확실] F1 mean이 R2보다 높은지와 paired fold delta의 방향을 보고한다. Mean이 올라도 일부 fold 악화·F1 std 증가·Recall/Precision 저하가 있으면 tradeoff를 명시한다. Mean 차이만으로 통계적 우위를 확정하지 않는다.
3. [확실] Parameter 증가0이라는 설계가 지켜졌는지 확인하고 실제 runtime/peak memory를 같은 정의로 보고한다. Loss behavior와 오류 분포를 함께 분석하되 낮은 loss로 낮은 F1을 덮지 않는다.
4. [확실] S3-1의 명확한 이점이 없으면 mean pooling을 채택하지 않고 R2를 유지한다. 모든 architecture를 차례로 자동 실행하지 않는다.
5. [확실] 이점이 있고 비용·tradeoff가 받아들일 만한지는 결과를 보고 사용자가 결정한다. 그때만 S3-2를 별도 설계·사전 고정하고, 현재는 두 번째 architecture를 확정하지 않는다.

## 12. Relationship to Behavior/Fusion

[확실] Context branch 출력은 기존과 같은 video/sequence-level [B,2] logits다. 필요 시 inference Softmax로 drowsy score를 제공하되 probability calibration이 보장됐다고 주장하지 않는다. 이후 EAR/MAR/Head Pose 기반 event와 연결할 수 있는 명확한 출력 단위를 유지한다.

[확실] Mean pooling은 frame-level EAR/blink detector나 알람 지속 규칙이 아니다. 현재20frames가 영상 전체에서 sampling되므로 offline clip 분류 성능을 실시간 조기 경보 성능으로 해석하지 않는다. Timestamp alignment, event aggregation, calibration, fusion rule과 latency 평가는 이후 별도 설계한다.

[확실] Behavior/Fusion은 NOT STARTED다. 이 실험에 EAR/MAR/pose, alert threshold, fusion weighting을 넣지 않는다. 간단한 Context 변경을 먼저 분리해 후속 Fusion에서 context 개선과 behavior 기여를 구분한다.

## 13. Experiment Roadmap

| ID | Scope | Additional training | Gate |
|---|---|---|---|
| **S3-0** | Existing R2 reference | **0** | Output/조건/상태 provenance 확인; 새 실행으로 부르지 않음 |
| **S3-1** | Mean hidden-state pooling only — IMPLEMENTED | **향후4folds×30epochs; 현재 NOT RUN** | Synthetic/meta targeted tests 통과; 실데이터 CUDA smoke/full training 미실행 |
| **S3-2** | Conditional second context experiment | **미정** | S3-1이 가치 있을 때만 별도 설계; 후보 조합 자동 실행 금지 |

[확실] **NEXT:** 구현 검토 후 사용자 요청으로 CUDA/실데이터 bounded smoke 또는 official Fold1을 별도 실행한다. 이번에는 full training과 실제 dataset 처리를 하지 않았다. R3/R5 재개는 별도 결정이다.

## 14. Status

| Work | Status |
|---|---|
| Stage2 Paper Reconstruction Refinement | **COMPLETED — scope closed** |
| R0 / R1 / R2 | COMPLETED |
| R3 | OPTIONAL / NOT RUN |
| R4 | DIAGNOSTIC COMPLETED / MIXED / NO NEW TRAINING RECOMMENDED |
| R5 | DEFER / NOT RUN |
| Stage3 Proposed Context Model Improvement | **IN PROGRESS** |
| S3-0 reference selection | R2 epoch30 official; existing artifacts |
| S3-1 architecture/config/script | **IMPLEMENTED** |
| S3-1 training/results | **NOT RUN / NOT AVAILABLE** |
| Behavior / Fusion | **NOT STARTED / NOT STARTED** |

[확실] Implementation snapshot: 전용 `stage3_temporal_mean.py`, `stage3_s3_1_temporal_mean.yaml`, `train_stage3_s3_1.py`를 생성했다. LSTM 옵션과 공용 checkpoint/environment의 condition ID만 확장했으며 기존 R2 semantics는 회귀 테스트로 확인했다. Attention/BiLSTM/Bottleneck/Behavior/Fusion은 구현하지 않았다.

[확실] Parameter count는 meta-device의 실제 VGG19 구조에서 total132,523,778 / trainable19,174,082 / frozen113,349,696으로 R2와 같았다. 작은 VGG-compatible mock에 synthetic [1,20,3,224,224]를 넣어 hidden[1,20,512]→mean[1,512]→logits[1,2]와 gradient policy를 확인했다. Targeted tests35개가 통과했으며 pretrained 다운로드/실제 영상 decode/full training은 실행하지 않았다. 실제 GPU memory/runtime와 S3-1 성능은 NOT AVAILABLE이다.

[확실] MLflow experiment는 `context-model-improvement`, parent/condition은 `S3_1_temporal_mean`, child는 `fold_1`~`fold_4`다. Stage2와 목적을 구분하며 reference R2와 temporal_readout=mean을 tag로 남긴다. Windows cp949 종료 오류를 피하도록 전용 script stdout/stderr를 UTF-8로 설정했다.

[확실] Checkpoint는 epoch 완료 후 model/optimizer/epoch/history/config/fold/RNG/shuffle generator를 기존 atomic 방식으로 저장한다. Condition ID와 config를 검증해 R2 checkpoint를 S3 resume에 사용하는 것을 거부한다. Synthetic checkpoint에서 next update/RNG 복원과 잘못된 fold/readout/condition 거부를 테스트했다. 실제 CUDA multiprocess resume의 bitwise 동일성은 미검증이다.

[확실] 출력은 `outputs/context_model/` 안의 새 경로만 허용한다. 신규 학습은 fresh ImageNet/seed initialization이며 R2 학습 checkpoint로 warm-start하지 않는다. 다음은 사용자 실행용 명령이며 이번에 실행하지 않았다. MLflow 서버가 실행 중이고 출력 폴더가 미사용이어야 한다.

```powershell
.\.venv\Scripts\python.exe scripts/train_stage3_s3_1.py --config configs/context_model/stage3_s3_1_temporal_mean.yaml --fold 1 --output-root outputs/context_model/S3_1_temporal_mean_fold1_official
```

[확실] Resume할 때는 동일 S3-1 config와 `--fold 1 --resume-from <trusted-S3-checkpoint> --output-root <new-path-under-outputs/context_model>`을 사용한다. 같은 condition/config/fold의 저장 epoch+1부터 새 run에 기록한다. S3-1 training/result는 계속 NOT RUN/NOT AVAILABLE이다.
