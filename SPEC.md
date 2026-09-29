> 이 문서는 **Codex 구현의 단일 동작 기준(Source of Truth)** 이다. 데이터 촬영/수집 계획은 `docs/DATA_COLLECTION.md`, AI 개발 규칙은 `docs/AI_CONVENTIONS.md`를 참고한다.
>

# 1. 목표

사용자가 제출한 청소 인증 사진 1장을 분석해:

1. 쓰레받이가 존재하는지 확인
2. 사진이 요청된 `zone_id + checkpoint_id` 장소가 맞는지 확인
3. 해당 장소의 청소 상태를 검수
4. 쓰레기가 쓰레받이 내부에 있는지 확인
5. PASS / FAIL 반환

## 개발 단계 원칙

현재는 실제 학습 데이터가 아직 준비되지 않았다고 가정한다.

따라서 먼저 다음을 **코드로 완성**한다.

- 데이터셋 검증/로딩 구조
- Dustpan YOLO Scratch Training 코드
- Zone Reference Bank 생성 코드
- PatchCore Memory Bank 생성 코드
- Validation/Threshold 산출 코드
- AI 추론 모듈
- Zone Registry
- API
- 테스트
- 전체 Training Orchestrator

실제 데이터가 준비된 뒤 정해진 폴더에 넣고 Training Pipeline을 실행하면 학습/빌드된 AI 자산이 생성되도록 한다.

> **금지:** API 요청이나 서버 시작 시 학습을 자동 실행하지 않는다.
>

---

# 2. 입력

## API 입력

- `image`: 청소구역과 쓰레받이가 함께 보이는 이미지 1장
- `zone_id`: 검수 대상 청소구역 ID
- `checkpoint_id`: 해당 Zone 내부 촬영/검수 단위 ID
- `user_id`: 선택값. AI 판정에는 사용하지 않는다.

## 식별 단위

```
zone_id
→ 청소구역 전체

checkpoint_id
→ 사진 1장으로 검수 가능한 최소 장소 단위
```

예:

```
main_stair_a
├── f1_f2
├── f2_f3
└── f3_f4
```

---

# 3. 전체 파이프라인

```
image + zone_id + checkpoint_id
          ↓
Image Validator
          ↓
Global Model Readiness Check
          ↓
Dustpan YOLO
          ├─ dustpan 없음 → DUSTPAN_NOT_FOUND / 즉시 FAIL
          └─ dustpan + trash 탐지
          ↓
Dustpan Mask
          ↓
Zone Registry Lookup
          ├─ zone 없음 → UNKNOWN_ZONE
          ├─ checkpoint 없음 → UNKNOWN_CHECKPOINT
          └─ Zone Asset 경로 확인
          ↓
Zone Asset Readiness Check
          ├─ Reference Bank 없음 → ZONE_MODEL_NOT_READY
          └─ PatchCore Memory Bank 없음 → ZONE_MODEL_NOT_READY
          ↓
Zone Recognition
          ├─ requested zone/checkpoint 검증 실패 → ZONE_NOT_RECOGNIZED / 즉시 FAIL
          ↓
PatchCore
          ↓
Trash-Inside 판정
          ↓
FinalEvaluator
          ↓
PASS / FAIL JSON
```

---

# 4. Dustpan YOLO

## 목적

- `dustpan` 존재 확인
- `trash` 탐지
- Bounding Box 제공

## 구조

- Ultralytics YOLO Object Detection
- 클래스: `0: dustpan`, `1: trash`
- pretrained `.pt` Weight 사용 금지
- YOLO 구조 `.yaml`에서 모델을 생성한 뒤 자체 데이터로 Scratch Training

## Training Pipeline

실제 데이터가 없어도 다음 코드 구조를 먼저 구현한다.

- dataset YAML 생성/검증
- label 검증
- scratch model 생성
- train
- validation
- best weight 저장
- metrics/metadata 저장

권장 명령:

```bash
python -m training.run_pipeline --component dustpan
```

데이터나 라벨이 없으면 `DATASET_NOT_READY`로 종료하고 가짜 Weight를 만들지 않는다.

---

# 5. Dustpan Mask

Dustpan YOLO가 반환한 Bounding Box를 검증한 뒤 전체 이미지에서 해당 영역만 Mask한다.

- Crop하지 않는다.
- 전체 크기와 구도를 유지한다.
- Zone Recognition과 PatchCore가 같은 `region_masker.py`를 사용한다.
- Reference 생성/PatchCore Build/실제 추론에서 동일한 Mask 규칙을 사용한다.
- 유효하지 않은 Bounding Box → `INVALID_DETECTION_RESULT`

---

# 6. Zone Recognition

## 목적

사진이 요청된 `zone_id + checkpoint_id` 장소가 맞는지 확인한다.

## 구조

고정 `zone_A / zone_B / unknown` Classifier를 사용하지 않는다.

MVP는 **Reference Embedding 기반 검증**을 사용한다.

```
Masked Image
→ Visual Encoder
→ L2-normalized Embedding
→ Requested Zone/Checkpoint Reference Bank와 Cosine Similarity
→ Threshold 검증
→ Recognized / Not Recognized
```

## Visual Encoder

- MVP 기본 Encoder는 **torchvision ResNet18 pretrained ImageNet weights**를 사용한다.
- 마지막 classification head를 사용하지 않고 feature embedding을 추출한 뒤 L2 Normalize한다.
- Encoder 구현은 interface/factory로 감싸 나중에 다른 Encoder로 교체 가능하게 한다.
- 전역 모델로 관리하며 새 Zone 추가 시 Encoder 재학습은 하지 않는다.
- 정확한 torchvision/PyTorch 버전과 Weight 식별자는 의존성 및 metadata에 고정한다.
- 런타임 인터넷 다운로드에 의존하지 않도록 모델 자산 준비 단계에서 Weight 사용 가능 여부를 확인한다.

## Reference Bank

각 `zone_id + checkpoint_id`별 Reference Embedding Bank를 생성한다.

새 Zone 추가:

```
Reference 이미지
→ 공통 Mask/Preprocess
→ Encoder
→ Embedding
→ Reference Bank 저장
→ Validation으로 Threshold 결정
→ Registry 등록
```

기존 Zone 전체를 다시 학습하지 않는다.

## Validation

- 해당 장소 positive 이미지
- 다른 Zone/Checkpoint 또는 미등록 장소 negative 이미지

를 사용해 `ZONE_RECOGNITION_THRESHOLD`를 검증한다.

## 판정

- similarity >= threshold → Recognized
- similarity < threshold → `ZONE_NOT_RECOGNIZED`

---

# 7. PatchCore

## 목적

올바른 장소의 청소 상태가 정상과 다른지 탐지한다.

## 데이터

각 `zone_id + checkpoint_id`별 별도 관리.

- Train: 정상만
- Validation: 정상 + 이상
- Test: 정상 + 이상

## Build Pipeline

```
정상 Train 이미지
→ Dustpan Mask
→ 동일 Resize/Normalize
→ Feature extraction
→ Memory Bank
→ Validation
→ anomaly threshold
→ 자산 + metadata 저장
```

권장 명령:

```bash
python -m training.run_pipeline --component patchcore --zone-id main_stair_a --checkpoint-id f1_f2
```

- `anomaly_score <= threshold` → 정상
- `anomaly_score > threshold` → `ZONE_ANOMALY_DETECTED`

PatchCore pretrained feature backbone 사용은 허용한다.

---

# 8. Zone Registry

개별 청소구역을 Python 조건문으로 하드코딩하지 않는다.

Registry 최소 정보:

```yaml
zone_id: main_stair_a
display_name: 본관 계단 A
active: true
checkpoints:
  f1_f2:
    reference_bank: models/zones/main_stair_a/f1_f2/reference_embeddings
    patchcore: models/zones/main_stair_a/f1_f2/patchcore
    zone_threshold: null
    anomaly_threshold: null
```

새 Zone/Checkpoint 추가:

1. Registry 항목 추가
2. 데이터 배치
3. Training Pipeline 실행
4. 생성된 metadata/threshold 반영

---

# 9. 전체 Training Orchestrator

권장 진입점:

```bash
python -m training.run_pipeline --component all
```

기능:

1. Dataset 구조 검증
2. YOLO 데이터/라벨 검사
3. YOLO Training 또는 기존 유효 Weight 사용
4. Registry Zone/Checkpoint 순회
5. Reference Bank 생성
6. PatchCore Memory Bank 생성
7. Validation
8. Threshold 저장
9. Metadata 저장
10. 결과 요약

규칙:

- 데이터가 없는 Component는 가짜 결과를 만들지 않는다.
- 부족한 데이터를 명확히 보고한다.
- 부분 실행을 지원한다.
- 유효한 기존 자산은 `--force` 없이는 덮어쓰지 않는다.
- Training Pipeline은 API 서버와 분리한다.

---

# 10. 권장 구조

```
app/
├── main.py
├── api/
│   └── cleaning.py
├── ai/
│   ├── dustpan_yolo.py
│   ├── region_masker.py
│   ├── zone_recognition.py
│   └── zone_anomaly.py
├── services/
│   ├── evaluator.py
│   └── zone_registry.py
├── schemas/
│   └── cleaning.py
├── core/
│   └── config.py
└── utils/
    └── image.py

training/
├── run_pipeline.py
├── dataset_validator.py
├── train_dustpan.py
├── build_zone_references.py
├── build_patchcore.py
└── validate_models.py

configs/
└── zones.yaml

datasets/
├── raw/
├── annotations/
├── splits/
└── processed/

models/
├── global/
│   ├── dustpan/
│   └── zone_encoder/
└── zones/
    └── <zone_id>/
        └── <checkpoint_id>/
            ├── reference_embeddings/
            ├── patchcore/
            └── metadata.json

tests/
```

---

# 11. Model Readiness

## 서버 시작

전역 자산 확인:

- Dustpan YOLO Weight
- Zone Visual Encoder

없거나 로드 실패 → `MODEL_NOT_READY`

## 요청 처리

요청된 `zone_id + checkpoint_id`:

- Reference Bank
- PatchCore Memory Bank
- Threshold/Metadata

누락/손상 → `ZONE_MODEL_NOT_READY`

한 Zone의 누락이 다른 Zone 서비스에 영향을 주지 않는다.

---

# 12. Early Return

다음은 이후 AI 단계를 실행하지 않는다.

1. 이미지 Decode 실패
2. 등록되지 않은 Zone
3. 등록되지 않은 Checkpoint
4. Dustpan 미검출
5. Zone Recognition 실패
6. 시스템/자산 오류

---

# 13. Trash-Inside 판정

초기 MVP:

- trash bbox center
- dustpan/trash intersection ratio

`TRASH_INSIDE_THRESHOLD`는 Validation으로 결정한다.

**여러 trash 중 하나라도 쓰레받이 외부면 `TRASH_OUTSIDE_DUSTPAN` FAIL.**

---

# 14. PASS 조건

모두 만족:

- 입력 정상
- 유효한 zone_id
- 유효한 checkpoint_id
- dustpan 검출
- Zone/Checkpoint recognition 성공
- PatchCore 정상
- trash 검출
- 모든 trash가 dustpan 내부

---

# 15. Error Code

## 검수 FAIL

- `DUSTPAN_NOT_FOUND`
- `ZONE_NOT_RECOGNIZED`
- `ZONE_ANOMALY_DETECTED`
- `TRASH_NOT_FOUND_IN_DUSTPAN`
- `TRASH_OUTSIDE_DUSTPAN`

## 입력 오류

- `INVALID_IMAGE`
- `UNKNOWN_ZONE`
- `UNKNOWN_CHECKPOINT`

## 시스템 오류

- `MODEL_NOT_READY`
- `ZONE_MODEL_NOT_READY`
- `INVALID_DETECTION_RESULT`
- `INFERENCE_ERROR`

## Training 오류

- `DATASET_NOT_READY`
- `INVALID_DATASET`
- `TRAINING_FAILED`

---

# 16. API 계약

```
POST /api/v1/cleaning/evaluate
```

`multipart/form-data`:

- `image`: required
- `zone_id`: required
- `checkpoint_id`: required
- `user_id`: optional

HTTP:

- `200`: 검수 완료 PASS/FAIL
- `400`: INVALID_IMAGE / UNKNOWN_ZONE / UNKNOWN_CHECKPOINT
- `413`: 업로드 제한 초과
- `415`: 지원하지 않는 이미지 형식
- `503`: MODEL_NOT_READY / ZONE_MODEL_NOT_READY
- `500`: INFERENCE_ERROR / INVALID_DETECTION_RESULT

---

# 17. Runtime 안전 규칙

- API 요청마다 Weight를 다시 로드하지 않는다.
- API 요청/서버 시작 시 학습하지 않는다.
- JPEG / PNG만 허용
- `MAX_UPLOAD_BYTES = 10MB`
- `MAX_IMAGE_PIXELS = 20,000,000`
- `MAX_INFERENCE_CONCURRENCY = 1`
- Bounding Box 범위 검증
- 추론 예외가 서버를 종료시키지 않음
- Python / PyTorch / Ultralytics / Anomalib / Zone Encoder 관련 버전을 검증 후 고정

---

# 18. 데이터가 없을 때의 완료 기준

실제 데이터 없이도 다음이 완료되어야 한다.

- 모든 모듈 import 가능
- FastAPI 실행 가능
- `/health` 동작
- Dataset Validator 구현
- Training Pipeline CLI 구현
- Registry 로딩/검증 구현
- YOLO Training 코드 구현
- Zone Reference 생성 코드 구현
- PatchCore Build 코드 구현
- Threshold/Metadata 저장 구조 구현
- 실제 Weight가 없을 때 올바른 MODEL_NOT_READY 응답
- Fixture/Mock으로 orchestration 로직 테스트
- **가짜 Weight나 실제처럼 보이는 가짜 모델 결과는 생성하지 않음**

실제 데이터가 준비되면 코드를 다시 작성하지 않고 Dataset을 넣고 Training Pipeline을 실행한다.
