# EcoGuard AI V1 명세

## 목적과 범위

V1은 사진 한 장을 공식 Gemini API에 보내 시각 검수 결과를 반환하는 MVP다. 직접 모델 학습, 학습 자산, 장소 인식은 포함하지 않는다. zone_id는 요청과 응답에 담는 기록용 Metadata이며 Gemini가 장소를 검증하지 않는다. checkpoint_id와 user_id는 V1 API 계약에 포함하지 않는다.

## 요청 흐름

1. FastAPI가 multipart/form-data 요청을 받는다.
2. 파일 크기, MIME 형식, 실제 이미지 디코딩 및 픽셀 수를 확인한다.
3. Gemini Service가 이미지와 고정 프롬프트를 공식 Google Gen AI Python SDK로 전송한다.
4. Gemini에 Pydantic 기반 Structured Output을 요청한다.
5. 응답을 CleaningAssessment로 다시 검증한다.
6. 유효한 결과를 Metadata와 함께 JSON으로 반환한다.

zone_id는 Gemini 입력에 포함하지 않는다. 서버는 실제 장소 목록이나 Zone Registry를 조회하지 않는다.

## 모델 선택과 한도 대응

- GEMINI_MODEL을 기본 모델로 사용한다. GEMINI_FALLBACK_MODELS는 쉼표로 구분한 대체 모델 목록이다. V1 기본 구성은 gemini-3.5-flash-lite와 gemini-3.1-flash-lite 두 모델만 사용한다.
- API 요청마다 두 모델 중 시작 모델을 번갈아 선택한다.
- HTTP 429면 현재 모델의 요청 한도 초과로 보고 즉시 다른 모델을 시도한다.
- HTTP 503이면 같은 모델을 1초 뒤 한 번 재시도한다. 다시 503이면 다른 모델을 시도한다.
- 한 모델에서 성공하면 해당 결과를 반환하고 다른 모델은 호출하지 않는다. 두 모델 모두 사용할 수 없으면 마지막 오류를 시스템 오류로 반환한다.
- 프로젝트 전체 한도나 지출 한도는 모델 전환으로 늘어나지 않는다.

## 판정 필드

- decision: PASS, FAIL 또는 REVIEW
- image_assessable: PASS/FAIL에 필요한 모든 조건을 사진에서 확실하게 판단할 수 있는지. 하나라도 불확실하면 false
- dustpan_detected: 쓰레받이가 보이는지
- trash_detected: 사진에 쓰레기가 보이는지
- trash_inside_dustpan: 보이는 쓰레기가 모두 쓰레받이 안에 있는지. 쓰레기가 전혀 없으면 false로 기록한다.
- cleaning_area_clean: 보이는 청소구역에 눈에 띄는 쓰레기나 오염이 남아 있지 않은지
- needs_review: 사람이 다시 확인해야 하는지. REVIEW와 함께 true여야 한다.
- reasons: 판단 근거를 짧게 설명하는 문자열 목록

PASS는 아래 조건이 모두 true일 때만 허용한다.

- image_assessable == true
- dustpan_detected == true
- trash_detected == true
- trash_inside_dustpan == true
- cleaning_area_clean == true

FAIL은 image_assessable == true이고 아래 조건 중 하나 이상이 false인 경우다.

- dustpan_detected
- trash_detected
- trash_inside_dustpan
- cleaning_area_clean

따라서 쓰레기가 보이지 않으면 trash_detected == false이므로 반드시 FAIL이다. 쓰레기가 모두 쓰레받이 안에 있고 청소구역이 깨끗하다는 점을 함께 인증해야 PASS다.

REVIEW는 사진 자체를 판정할 수 없거나 가림, 흐림, 어두움 등으로 위 조건 중 하나 이상을 확실히 판단할 수 없는 경우다. 스키마에는 조건별 confidence 필드가 없으므로, 필수 조건 하나라도 불확실하면 image_assessable == false로 기록하고 decision=REVIEW, needs_review=true를 설정한다. image_assessable == true이면 모든 조건을 확실히 판단한 상태여야 하며, 그 값에 따라 PASS 또는 FAIL이 결정된다.

## API 계약

### POST /api/v1/cleaning/evaluate

Content-Type은 multipart/form-data다.

필수 입력:

- image: JPEG, PNG 또는 WebP 사진 파일
- zone_id: 기록용 청소구역 식별자

zone_id는 공백이 아닌 1~128자 문자열이어야 한다. 기본 업로드 한도는 10 MiB, 이미지 해상도 한도는 20 megapixels다.

성공 응답은 HTTP 200이며 다음 JSON 필드를 포함한다.

- decision
- image_assessable
- dustpan_detected
- trash_detected
- trash_inside_dustpan
- cleaning_area_clean
- needs_review
- reasons
- zone_id

checkpoint_id와 user_id는 요청과 응답에 포함하지 않는다.

## 시스템 오류

시스템 오류는 PASS/FAIL/REVIEW 응답 대신 error_code와 message JSON을 반환한다.

- API Key 누락: HTTP 503, GEMINI_API_KEY_MISSING
- Gemini 호출 Timeout: HTTP 504, GEMINI_TIMEOUT
- Rate Limit: HTTP 503, GEMINI_RATE_LIMITED
- 기타 Gemini 호출 실패: HTTP 502, GEMINI_API_ERROR
- Structured Output 파싱 또는 Pydantic 검증 실패: HTTP 502, STRUCTURED_OUTPUT_INVALID
- 지원하지 않는 형식 또는 잘못된 이미지: HTTP 422, INVALID_IMAGE
- 업로드 크기 초과: HTTP 413, IMAGE_TOO_LARGE
- 필수 필드 누락 또는 zone_id 검증 실패: FastAPI HTTP 422

GET /health는 외부 Gemini 호출 없이 HTTP 200과 상태를 반환한다. Gemini API Key가 없더라도 서버 시작과 Health 확인은 가능하다.

## 보안 및 개인정보

- API Key는 GEMINI_API_KEY 환경변수에서 읽는다. 저장소에는 빈 값만 담은 .env.example을 둔다.
- .env, 촬영 데이터, 모델 산출물, 가상환경을 Git에 올리지 않는다.
- 사진 바이트와 API Key를 애플리케이션 로그에 남기지 않는다.
- AI 판정에는 이미지와 정적 검사 프롬프트만 전달한다. zone_id Metadata는 전송하지 않는다.

## V1 제외 항목

YOLO Training, PatchCore, Reference Embedding, 직접 Zone Recognition, Dataset Pipeline, Training Pipeline, 직접 모델 Weight 생성은 V2 범위다.
