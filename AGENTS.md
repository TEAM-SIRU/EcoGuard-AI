# EcoGuard AI V1 작업 규칙

## 목표

스마트폰 사진 한 장을 Gemini API로 검사해 청소 결과를 PASS, FAIL 또는 REVIEW로 반환하는 FastAPI MVP를 만든다. 사람이 확인해야 하는 사진은 억지로 PASS/FAIL로 확정하지 않고 REVIEW로 반환한다.

## V1 구조

사진 업로드 → FastAPI → Gemini API → Structured Output → Pydantic 검증 → PASS / FAIL / REVIEW JSON

Gemini는 사진 판독 가능 여부, 쓰레받이와 쓰레기의 존재 여부, 쓰레받이 안에 쓰레기가 있는지, 청소구역이 깨끗한지, 사람의 재확인이 필요한지를 평가한다.

## 규칙

- 직접 AI 모델을 학습하거나 모델 Weight를 만들지 않는다.
- YOLO, PatchCore, Reference Embedding, Zone Recognition 학습, Dataset 수집 및 Training Pipeline은 V2 범위다.
- Gemini API 호출은 service layer에서만 한다. 공식 Google Gen AI Python SDK와 Pydantic Structured Output을 사용한다.
- API Key와 모델명은 GEMINI_API_KEY, GEMINI_MODEL 환경변수로 설정한다. 비밀 키를 코드나 Git에 넣지 않는다.
- zone_id와 checkpoint_id는 요청과 응답에 기록하는 Metadata다. Gemini에 보내거나 실제 장소 검증에 사용하지 않는다.
- user_id는 선택 Metadata이며 AI 판정에 사용하지 않는다.
- Gemini API 오류, Key 누락, Timeout, Rate Limit, Structured Output 검증 실패는 시스템 오류다. 청소 FAIL로 바꾸지 않는다.
- 사진을 판정하기 어렵거나 결과가 애매하면 REVIEW를 반환한다.
- API 요청 처리 중이나 서버 시작 시 AI 학습을 실행하지 않는다.
- 구현 기준은 SPEC.md, 작업 순서는 TASKS.md를 따른다.

## 변경 규칙

명세와 구현이 충돌하거나 PASS/FAIL 업무 기준을 바꿔야 하면 임의로 정책을 정하지 말고 결과에 OPEN ISSUE로 남긴다. V2 구현은 v2-custom-ai 브랜치에서 관리하며 main에 다시 추가하지 않는다.
