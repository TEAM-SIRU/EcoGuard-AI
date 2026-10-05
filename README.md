# EcoGuard AI V1

스마트폰 사진 한 장을 Gemini API로 검사해 청소 결과를 PASS, FAIL 또는 REVIEW로 반환하는 FastAPI MVP입니다. 직접 모델을 학습하지 않으며, 애매하거나 판정하기 어려운 사진은 REVIEW로 전달합니다.

## 동작 흐름

    사진 업로드 → FastAPI → Gemini API → Structured Output → Pydantic 검증 → JSON 응답

Gemini는 사진 판독 가능 여부, 쓰레받이/쓰레기 존재 여부, 쓰레기가 쓰레받이 안에 있는지, 청소구역이 깨끗한지, 사람의 재확인이 필요한지를 평가합니다. 요청에는 청소구역 식별용 zone_id만 함께 보내며, 이 값은 기록용 Metadata로만 사용하고 Gemini에 보내거나 실제 장소 검증에 사용하지 않습니다. checkpoint_id와 user_id는 V1 요청에서 받지 않습니다.

PASS에는 판정 가능, 쓰레받이 검출, 쓰레기 검출, 쓰레받이 안에 수거, 청소구역 깨끗함이 모두 필요합니다. 판정 가능한 사진에서 하나라도 충족하지 않으면 FAIL입니다. 필수 조건 중 하나라도 확실히 볼 수 없으면 REVIEW입니다. 쓰레기가 보이지 않는 사진은 PASS가 될 수 없습니다.

## 설치

Python 3.11 이상 3.14 미만을 사용합니다.

    py -3.11 -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    Copy-Item .env.example .env

.env 파일을 열고 Google AI Studio에서 발급한 키를 GEMINI_API_KEY에 설정합니다. 기본 모델은 이미지 입력과 구조화 응답을 지원하는 고처리량 모델 gemini-3.5-flash-lite이며, GEMINI_MODEL 환경변수로 변경할 수 있습니다. .env는 Git에 포함되지 않습니다.

현재 두 Lite 모델만 사용합니다: 기본 gemini-3.5-flash-lite와 대체 gemini-3.1-flash-lite입니다. 요청마다 시작 모델을 번갈아 선택합니다. HTTP 429(요청 한도) 응답이면 즉시 다른 Lite 모델로 전환하고, HTTP 503(일시적 이용 불가)이면 같은 모델을 한 번 재시도한 뒤 다른 모델로 전환합니다. 두 모델 모두 사용할 수 없으면 오류를 반환합니다. 모델별 한도는 다를 수 있지만, 프로젝트 전체 한도는 모델을 바꿔도 해소되지 않습니다. 현재 한도는 [Google AI Studio의 Rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)에서 확인하세요.

## 서버 실행

    .venv\Scripts\Activate.ps1
    python -m uvicorn app.main:app --reload

서버 확인은 GET /health, API 문서는 /docs에서 할 수 있습니다. Health 확인은 Gemini API를 호출하지 않습니다.

## API 사용

POST /api/v1/cleaning/evaluate에 multipart/form-data를 보냅니다.

필수 필드:
- image: JPEG, PNG 또는 WebP 파일
- zone_id: 기록용 청소구역 식별자(공백이 아닌 1~128자 문자열)

PowerShell 예시:

    curl.exe -X POST http://127.0.0.1:8000/api/v1/cleaning/evaluate -F "image=@stairs.jpg" -F "zone_id=zone-a"

정상 응답에는 decision, 이미지 평가 필드, reasons, 요청에서 받은 zone_id가 포함됩니다. decision 값은 PASS, FAIL 또는 REVIEW입니다. checkpoint_id와 user_id는 요청 및 응답에 포함되지 않습니다. Key 누락이나 Gemini 오류는 청소 판정과 별도의 error_code/message JSON과 HTTP 오류 상태로 반환합니다.

Gemini가 일시적으로 요청을 거절하는 HTTP 429 또는 503 오류에는 최대 2회 재시도하며, 간격은 1초와 2초입니다. 재시도 후에도 실패하면 화면에 오류 코드가 표시됩니다.

기본 업로드 크기는 10 MiB, 이미지 해상도는 20 megapixels까지 허용합니다. 환경변수 MAX_UPLOAD_BYTES와 MAX_IMAGE_PIXELS로 바꿀 수 있습니다.

## 테스트

Unit Test는 Gemini Client를 Mock하고 실제 API Key나 네트워크 호출 없이 실행합니다.

    python -m pytest

## V1 제외 범위

YOLO Training, PatchCore, Reference Embedding, 직접 Zone Recognition, Dataset 수집 Pipeline, Training Pipeline 및 직접 모델 Weight 생성은 V2 범위이며 v2-custom-ai 브랜치에서 관리합니다.
