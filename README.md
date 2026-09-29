# EcoGuard-AI
환경지킴이 AI 모델 레포지토리입니다.
# ai 작동방식 워크플로
사용자

   │
   ├──────── 청소구역 사진
   │
   └──────── 쓰레받이 사진
             │
             ▼
        Backend / AI API
             │
      ┌──────┴───────┐
      │              │
      ▼              ▼
청소구역 AI        쓰레받이 AI

PatchCore          YOLO
이상탐지           Object Detection

      │              │
      ▼              ▼
이상 영역         dustpan
존재 여부         trash
      │              │
      └──────┬───────┘
             ▼
        판정 알고리즘
             │
             ▼
         PASS / FAIL
             │
             ▼
            JSON