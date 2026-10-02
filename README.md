# EcoGuard AI

스마트폰 사진 1장으로 쓰레받이, 요청한 Zone/Checkpoint, 청소 상태를 검수하는 FastAPI와 별도 AI 자산 Build 파이프라인입니다. API는 학습을 실행하지 않으며, 필요한 실제 자산이 없으면 `MODEL_NOT_READY` 또는 `ZONE_MODEL_NOT_READY`를 반환합니다.

## AI workflow

```mermaid
flowchart TD
    A[One photo: stairs + dustpan] --> B[Full-image YOLO finds dustpan]
    B --> C[Split into Stair View and Dustpan Crop]
    C --> D[Zone Recognition on Stair View]
    D --> E[PatchCore on Stair View; dustpan feature ROI ignored]
    E --> F[YOLO trash check on Dustpan Crop]
    F --> G[FinalEvaluator: PASS / FAIL]
```

## 개발 환경

Python 3.11을 사용합니다.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Torch/Torchvision은 `requirements.txt`에 호환 버전으로 고정했습니다. Zone Encoder와 PatchCore는 ImageNet ResNet18 weight를 런타임에 다운로드하지 않습니다. 검증된 torchvision ResNet18 weight 파일을 다음 경로에 사전 배치해야 합니다.

```text
models/global/zone_encoder/resnet18-f37072fd.pth
```

Dustpan weight는 Training Pipeline이 실제 데이터로 생성하는 `models/global/dustpan/best.pt`입니다. 저장소는 어떤 가중치, 데이터, 추론 결과도 제공하지 않습니다.

## 서버

```powershell
uvicorn app.main:app --reload
```

- `GET /health`
- `POST /api/v1/cleaning/evaluate` (`multipart/form-data`: `image`, `zone_id`, `checkpoint_id`, optional `user_id`)

`/health`는 모델 자산이 없어도 200으로 서버 상태를 알리고, `models_ready`와 누락 자산을 함께 반환합니다. 청소 평가 요청은 JPEG/PNG 이미지 형식, 10 MB 업로드, 20 MP 픽셀 제한을 적용합니다.

경로와 런타임 제한은 환경 변수로 조정할 수 있습니다: `ECOGUARD_ROOT`, `ECOGUARD_DATASET_DIR`, `ECOGUARD_MODEL_DIR`, `ECOGUARD_ZONE_REGISTRY`, `ECOGUARD_DUSTPAN_WEIGHT`, `ECOGUARD_ZONE_ENCODER_WEIGHT`, `ECOGUARD_MAX_UPLOAD_BYTES`, `ECOGUARD_MAX_IMAGE_PIXELS`, `ECOGUARD_MAX_INFERENCE_CONCURRENCY`, `ECOGUARD_IMAGE_SIZE`.


## Zone Registry

`configs/zones.yaml`에 Zone/Checkpoint를 등록합니다. 초기 Registry는 비어 있습니다.

```yaml
zones:
  main_stair_a:
    display_name: 본관 계단 A
    active: true
    checkpoints:
      f1_f2:
        reference_bank: models/zones/main_stair_a/f1_f2/reference_embeddings
        patchcore: models/zones/main_stair_a/f1_f2/patchcore
        zone_threshold: null
        anomaly_threshold: null
```

Threshold는 Training Pipeline에서 분리된 positive/negative validation 점수로 계산하여 Checkpoint `metadata.json`에 보관합니다. Registry의 threshold 항목은 사람이 확인하기 위한 초기 null 필드이며, API 런타임 값은 생성 metadata에서 읽습니다.

## Dataset 배치

Images may be JPG/JPEG/PNG. Keep capture sessions separate between train and validation. Zone reference builds mask a detected pan with Dustpan YOLO. PatchCore training uses stair-only photos directly and does not require a trained dustpan model.

```text
datasets/raw/
├── dustpan/
│   ├── images/{train,val}/<image>.jpg
│   └── labels/{train,val}/<image>.txt
├── zones/<zone_id>/<checkpoint_id>/
│   ├── reference/<image>.jpg
│   └── validation/positive/<image>.jpg
├── zones/negative/<zone_id>/<checkpoint_id>/<image>.jpg
└── patchcore/<zone_id>/<checkpoint_id>/
    ├── train/normal/<image>.jpg
    └── val/{normal,anomaly}/<image>.jpg
```

Dustpan YOLO label은 이미지와 같은 basename의 UTF-8 `.txt`이며 각 줄은 `class_id x_center y_center width height` normalized 값입니다(0=dustpan, 1=trash). 빈 label은 객체 없는 YOLO 학습 사진에 허용됩니다.

Zone reference/validation and PatchCore images show the stair scene. Zone negative photos must show another or unregistered place. PatchCore trains on normal photos and ordinarily validates with held-out normal plus real anomaly photos. If real anomaly photos are not available, use `--normal-only-threshold` only as a provisional threshold: it uses the highest held-out normal score and does not validate anomaly detection. PatchCore training runs independently from Dustpan YOLO. At serving, the detected pan bbox and a 32px context area are excluded from PatchCore anomaly scoring.


## Independent training components

- `--component patchcore` builds only the per-checkpoint stair anomaly model from `datasets/raw/patchcore/<zone>/<checkpoint>/`. It uses the local ResNet18 encoder and does not load `models/global/dustpan/best.pt`.
- `--component dustpan` separately trains the scratch YOLO detector from `datasets/raw/dustpan/` and its labels.
- PatchCore validation with real anomalies reports balanced accuracy and score ranges. Normal-only mode records `anomaly_detection_validated: false`; replace that provisional threshold after adding real, held-out anomaly photos to `val/anomaly`.

## Training / Build

```powershell
python -m training.run_pipeline --component dustpan
python -m training.run_pipeline --component zone --zone-id main_stair_a --checkpoint-id f1_f2
python -m training.run_pipeline --component patchcore --zone-id zone_a --checkpoint-id stair_1
python -m training.run_pipeline --component patchcore --zone-id zone_b --checkpoint-id stair_1 --normal-only-threshold
python -m training.run_pipeline --component all
```

기존 유효 자산은 기본적으로 덮어쓰지 않습니다. 재생성하려면 `--force`를 지정합니다. Dataset/label 누락은 `DATASET_NOT_READY`, 형식 오류는 `INVALID_DATASET`으로 보고됩니다. Dustpan은 pretrained `.pt`가 아닌 Ultralytics YAML architecture에서 초기화하며, 위 명령을 직접 실행할 때만 학습합니다.

## 테스트

```powershell
python -m pytest
```

테스트는 작은 Fixture를 사용하며 실제 사진, 가중치, 가짜 추론 결과를 생성하지 않습니다.

## 범위 및 OPEN ISSUE

Dataset and model files are local ignored assets and are not checked into Git. The current workspace has a built local PatchCore asset for A/stair_1. Full API inference still requires a trained Dustpan YOLO weight and Zone Reference Bank.
