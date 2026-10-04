# EcoGuard AI V1

스마트폰 사진 한 장을 Gemini API로 검사해 청소 결과를 PASS, FAIL 또는 REVIEW로 반환하는 FastAPI MVP입니다. 직접 모델을 학습하지 않으며, 애매하거나 판정하기 어려운 사진은 REVIEW로 전달합니다.

## 동작 흐름

    사진 업로드 → FastAPI → Gemini API → Structured Output → Pydantic 검증 → JSON 응답

Gemini는 사진 판독 가능 여부, 쓰레받이/쓰레기 존재 여부, 쓰레기가 쓰레받이 안에 있는지, 청소구역이 깨끗한지, 사람의 재확인이 필요한지를 평가합니다. zone_id와 checkpoint_id는 기록용 Metadata로만 사용하며 Gemini에 보내지 않습니다.

## 설치

Python 3.11 이상 3.14 미만을 사용합니다.

    py -3.11 -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    Copy-Item .env.example .env

.env 파일을 열고 Google AI Studio에서 발급한 키를 GEMINI_API_KEY에 설정합니다. GEMINI_MODEL은 기본값 gemini-3.8-flash이며 환경변수로 변경할 수 있습니다. .env는 Git에 포함되지 않습니다.

## 서버 실행

    .venv\Scripts\Activate.ps1
    python -m uvicorn app.main:app --reload

서버 확인은 GET /health, API 문서는 /docs에서 할 수 있습니다. Health 확인은 Gemini API를 호출하지 않습니다.

## API 사용

POST /api/v1/cleaning/evaluate에 multipart/form-data를 보냅니다.

필수 필드:
- image: JPEG, PNG 또는 WebP 파일
- zone_id: 기록용 문자열
- checkpoint_id: 기록용 문자열

선택 필드:
- user_id: 기록용 문자열

PowerShell 예시:

    curl.exe -X POST http://127.0.0.1:8000/api/v1/cleaning/evaluate -F "image=@stairs.jpg" -F "zone_id=zone-a" -F "checkpoint_id=stair-1" -F "user_id=user-1"

정상 응답에는 decision, 이미지 평가 필드, reasons, 요청 Metadata가 포함됩니다. decision 값은 PASS, FAIL 또는 REVIEW입니다. Key 누락이나 Gemini 오류는 청소 판정과 별도의 error_code/message JSON과 HTTP 오류 상태로 반환합니다.

기본 업로드 크기는 10 MiB, 이미지 해상도는 20 megapixels까지 허용합니다. 환경변수 MAX_UPLOAD_BYTES와 MAX_IMAGE_PIXELS로 바꿀 수 있습니다.

## 테스트

Unit Test는 Gemini Client를 Mock하고 실제 API Key나 네트워크 호출 없이 실행합니다.

    python -m pytest

## V1 제외 범위

YOLO Training, PatchCore, Reference Embedding, 직접 Zone Recognition, Dataset 수집 Pipeline, Training Pipeline 및 직접 모델 Weight 생성은 V2 범위이며 v2-custom-ai 브랜치에서 관리합니다.
