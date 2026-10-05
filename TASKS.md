# EcoGuard AI V1 작업 순서

- [x] 기존 V2 작업을 v2-custom-ai 브랜치와 GitHub에 보존
- [x] main의 Gemini V1 명세 및 프로젝트 설명 작성
- [x] V1 의존성, 환경 설정, 비밀정보 제외 규칙 구성
- [x] CleaningAssessment Structured Output Schema 구현
- [x] 이미지 입력 검증과 FastAPI API 구현
- [x] Gemini Service 및 검사 프롬프트 구현
- [x] 시스템 오류와 청소 판정 결과 분리
- [x] Gemini 호출을 Mock한 테스트 작성
- [x] 쓰레받이와 수거한 쓰레기를 함께 인증하도록 최종 판정 정책 통일
- [x] 테스트 실행 결과 확인
- [x] main 변경사항 commit 및 가능한 경우 GitHub push

실제 Gemini API Key나 호출 없이 Unit Test를 실행한다. V2 범위는 이 작업에서 자동 진행하지 않는다.
