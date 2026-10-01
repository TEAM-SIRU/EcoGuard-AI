## 현재 목표

**실제 데이터 수집 전에 EcoGuard AI의 전체 코드 구조와 Training Pipeline을 구현한다.**

---

## Phase 0. 하네스/명세

- [x]  단일 이미지 구조 확정
- [x]  Zone / Checkpoint 구조 확정
- [x]  Reference Embedding 기반 Zone Recognition 방향 확정
- [x]  새 Zone 추가 시 기존 전체 모델 재학습을 피하는 구조 확정
- [x]  AI 학습/추론 규칙 정리
- [x]  데이터 수집 장소 목록 정리

## Phase 1. 프로젝트 부트스트랩

- [x]  권장 폴더 구조 생성
- [x]  Python 가상환경/의존성 파일 구성
- [x]  FastAPI 기본 앱 + `/health`
- [x]  `config.py` 구조 생성
- [x]  Request / Response / Error Schema 생성
- [x]  Zone Registry 설정 파일/Loader 구조 생성
- [x]  기본 테스트 환경 구성
- [x]  `.gitignore` 구성

## Phase 2. 데이터 없이 AI 구조 선구현

### 공통

- [x]  Image Validator 구현
- [x]  `region_masker.py` 구현
- [x]  Dataset Validator 구현
- [x]  모델/자산 Metadata Schema 구현
- [x]  Threshold 저장/로드 구조 구현

### Dustpan YOLO

- [x]  Scratch YOLO 모델 생성 코드
- [x]  Dataset YAML/Label 검증 코드
- [x]  Training 함수/CLI 구현
- [x]  Validation/Best Weight 저장 구조 구현
- [x]  실제 Weight 없을 때 MODEL_NOT_READY 처리
- [x]  추론 Wrapper 구현

### Zone Recognition

- [x]  Visual Encoder Loader 구현
- [x]  Stair View Embedding extraction
- [x]  L2 Normalize + Cosine Similarity
- [x]  Zone/Checkpoint별 Reference Bank 생성
- [x]  Reference Bank 저장/로드
- [x]  Positive/Negative Validation 구조
- [x]  ZONE_RECOGNITION_THRESHOLD 저장/로드
- [x]  새 Zone 추가 시 Encoder 재학습 없이 Reference Bank만 추가 가능한지 테스트

### PatchCore

- [x]  PatchCore Builder 구조 구현
- [x]  Zone/Checkpoint별 정상 데이터 Loader
- [x]  Shared Stair View preprocessing
- [x]  Memory Bank 생성/저장/로드
- [x]  Validation으로 anomaly threshold 결정하는 구조
- [x]  자산 누락 시 ZONE_MODEL_NOT_READY 처리

### Training Orchestrator

- [x]  `training/run_pipeline.py` 구현
- [x]  `--component dustpan|zone|patchcore|all`
- [x]  `--zone-id`, `--checkpoint-id`
- [x]  데이터 부족 시 DATASET_NOT_READY
- [x]  기존 자산 보호 + `--force`
- [x]  Metadata/결과 요약

## Phase 3. API / Evaluator 선구현

- [x]  `POST /api/v1/cleaning/evaluate`
- [x]  `image + zone_id + checkpoint_id`
- [x]  Global Model Readiness Check
- [x]  Zone Asset Readiness Check
- [x]  Dustpan Early Return
- [x]  Split image into Stair View and Dustpan Crop
- [x]  Zone Recognition Early Return
- [x]  PatchCore 연결
- [x]  Run YOLO trash detection on Dustpan Crop
- [x]  FinalEvaluator
- [x]  PASS/FAIL JSON
- [x]  시스템 오류 JSON
- [x]  Weight 없는 상태의 503 테스트

## Phase 4. 실제 데이터 수집/정리

### Dustpan

- [ ]  원본 수집
- [ ]  Check trash on stairs with PatchCore anomaly detection
- [ ]  Train / Validation / Test 분리

### Zone / Checkpoint

- [ ]  촬영 목록 기준 데이터 수집
- [ ]  zone_id / checkpoint_id 기록
- [ ]  Reference Positive 데이터 정리
- [ ]  다른 Zone/장소 Negative Validation 데이터 정리
- [ ]  촬영 세션 단위 Split

### PatchCore

- [ ]  정상 Train
- [ ]  정상/이상 Validation
- [ ]  정상/이상 Test

## Phase 5. 실제 Training / Asset Build

- [ ]  Dustpan YOLO 학습
- [ ]  Dustpan Validation
- [ ]  Best Weight 확정
- [ ]  전체 Reference Bank 생성
- [ ]  Zone Recognition threshold 검증
- [ ]  전체 PatchCore Memory Bank 생성
- [ ]  anomaly threshold 검증
- [ ]  Metadata/Version 저장

## Phase 6. End-to-End 검증

- [ ]  정상 PASS
- [ ]  쓰레받이 없음
- [ ]  잘못된 Zone
- [ ]  잘못된 Checkpoint
- [ ]  다른 장소
- [ ]  Check trash on stairs with PatchCore anomaly detection
- [ ]  Check trash on stairs with PatchCore anomaly detection
- [ ]  청소구역 anomaly
- [ ]  잘못된 이미지
- [ ]  YOLO Weight 누락
- [ ]  Reference Bank 누락
- [ ]  PatchCore Memory Bank 누락
- [ ]  업로드 제한
- [ ]  동시 요청 제한
- [ ]  실제 현장 테스트

## Codex 작업 규칙

- 실제 데이터가 없다고 구현을 멈추지 않는다.
- 데이터가 필요한 부분은 **재현 가능한 Training/Build Pipeline과 명확한 입력 구조까지 구현**한다.
- 가짜 Weight, 가짜 Metrics, 실제처럼 보이는 가짜 추론 결과를 만들지 않는다.
- 데이터가 없어서 Training을 실행할 수 없으면 `DATASET_NOT_READY`로 보고한다.
- 한 번에 한 Phase 또는 명확한 기능 단위만 작업한다.
- 작업 후 테스트 결과와 OPEN ISSUE를 보고한다.
- 사용자 승인 없이 다음 Phase로 자동 진행하지 않는다.
