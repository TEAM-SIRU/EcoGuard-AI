## Project Goal

EcoGuard AI는 스마트폰 사진 **1장**으로 지정된 청소구역/Checkpoint와 쓰레받이를 검수하는 AI API다.

최종 검수 결과는 `PASS / FAIL`, `is_passed`, 실패 사유를 반환한다.

## Architecture

1. **Dustpan YOLO**: `dustpan`, `trash` 탐지
2. **Region Masker**: 검출된 쓰레받이 영역 Mask
3. **Zone Recognition**: 요청된 `zone_id + checkpoint_id`의 장소가 맞는지 검증
4. **PatchCore**: 해당 Zone/Checkpoint의 청소 상태 이상탐지
5. **FinalEvaluator**: AI 결과를 규칙 기반으로 종합

## Current Development Strategy

- **실제 데이터 수집 전에도 전체 코드 구조와 학습/빌드 파이프라인을 먼저 구현한다.**
- 실제 데이터가 없어도 프로젝트가 import/test 가능한 상태를 만든다.
- 가짜 모델 Weight나 가짜 추론 결과를 생성하지 않는다.
- 학습 자산이 없으면 실제 추론 API는 `MODEL_NOT_READY` 또는 `ZONE_MODEL_NOT_READY`를 반환한다.
- 나중에 데이터를 정해진 Dataset 구조에 넣고 학습 파이프라인을 실행하면 필요한 Weight, Reference Bank, PatchCore Memory Bank, Threshold/Metadata가 생성되도록 한다.
- **API 요청 중이나 서버 시작 시 자동 학습하지 않는다.** 학습은 별도의 training pipeline에서 명시적으로 실행한다.

## Main Rules

- 기본 입력은 `image`, `zone_id`, `checkpoint_id`다.
- `user_id`는 필요할 경우에만 선택적으로 받으며 AI 판정에는 사용하지 않는다.
- 쓰레받이 미검출 또는 장소 검증 실패 시 즉시 FAIL하고 이후 AI 검수를 중단한다.
- Dustpan YOLO는 Ultralytics 구조를 사용하고 pretrained `.pt` 없이 자체 데이터로 Scratch Training한다.
- Zone Recognition은 고정 Classifier 대신 **Reference Embedding 기반 검증 구조**를 사용해 새 Zone 추가 시 기존 전체 모델 재학습을 피한다.
- Zone Recognition의 기본 Encoder는 프로젝트에서 고정한 pretrained visual encoder를 사용하며, Reference Bank는 Zone/Checkpoint별 데이터로 생성한다.
- PatchCore는 `zone_id + checkpoint_id`별 정상 데이터를 사용해 Memory Bank를 생성한다.
- 쓰레받이 영역은 Crop하지 않고 Mask 처리한 뒤 Zone Recognition/PatchCore에 전달한다.
- 학습/빌드와 추론의 Mask/Resize/Normalize 규칙은 동일하게 유지한다.
- 개별 Zone을 Python `if/elif`로 하드코딩하지 않는다. Zone Registry와 데이터/모델 자산으로 확장한다.
- 모델 모듈은 분석 결과만 반환하고 최종 PASS/FAIL은 `FinalEvaluator`가 결정한다.
- Training 코드와 API Serving 코드를 분리한다.
- 시스템 오류와 실제 청소 FAIL을 구분한다.
- 구현 동작 기준은 `SPEC.md`, AI 규칙은 `docs/AI_CONVENTIONS.md`, 현재 작업은 `TASKS.md`를 따른다.

## Change Rule

- 명세와 구현이 충돌하면 임의로 설계를 바꾸지 않는다.
- API 계약, AI 구조, Zone/Checkpoint 규칙을 변경해야 한다면 작업 결과에 OPEN ISSUE로 보고하고 사용자 확인 후 변경한다.