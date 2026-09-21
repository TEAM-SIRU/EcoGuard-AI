# EcoGuard-AI
환경지킴이 AI 모델 레포지토리입니다.
# ai 작동방식 워크플로
```mermaid
flowchart TD
    A[사용자]

    A --> B[청소구역 사진]
    A --> C[쓰레받이 사진]

    B --> D[Backend / AI API]
    C --> D

    D --> E[청소구역 AI]
    D --> F[쓰레받이 AI]

    E --> G[PatchCore<br/>이상 탐지]
    F --> H[YOLO<br/>Object Detection]

    G --> I[이상 영역<br/>존재 여부]
    H --> J[dustpan / trash<br/>탐지]

    I --> K[판정 알고리즘]
    J --> K

    K --> L{최종 판정}

    L -->|정상| M[PASS]
    L -->|이상| N[FAIL]

    M --> O[JSON Response]
    N --> O
    