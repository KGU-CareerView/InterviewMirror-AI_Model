# InterviewMirror-AI Model

## 프로젝트 개요

비대면 면접 환경에서 지원자의 표정을 실시간으로 분석하여
면접 상황에 특화된 3개 클래스로 분류하는 경량 AI 모델입니다.

---

## 클래스 정의

| ID  | 클래스  | 의미                        |
| --- | ------- | --------------------------- |
| 0   | Stable  | 안정적이고 자신감 있는 상태 |
| 1   | Nervous | 긴장하거나 불안한 상태      |
| 2   | Neutral | 감정 표현이 없는 중립 상태  |

---

## 모델 성능 (v3)

```text
Accuracy      : 89.6%
Macro F1      : 0.899
Nervous Recall: 97.1%
CPU 추론 속도  : 28ms (ONNX Runtime 기준)
```

---

## 레포지토리 구조

```text
InterviewMirror-AI_Model/
├── train/
│   ├── train_v3.py                 # v3 모델 학습 (2단계 전이학습)
│   ├── preprocess_affectnet.py     # AffectNet OpenCV 전처리
│   ├── preprocess_mediapipe.py     # FER+ MediaPipe 전처리
│   ├── merge_dataset.py            # 데이터 통합 + 클래스 균형
│   └── export_onnx.py              # ONNX 변환 + 자동 검증
├── results/
│   ├── v3_history.csv              # 에폭별 학습 이력
│   ├── f1_curve.png                # F1-Score 학습 곡선
│   ├── confusion_matrix.png        # Confusion Matrix
│   └── inference_speed.png         # 추론 속도 비교
├── docs/
│   └── MODEL_SPEC.md               # 모델 입출력 스펙
└── requirements.txt
```

---

## 데이터셋

```text
베이스 데이터셋 : AffectNet
추가 데이터셋  : FER+ (Stable, Neutral 보강용)
최종 학습 데이터: 24,118장
클래스 분포    : Stable 33.2% / Nervous 33.2% / Neutral 33.6%
```

---

## 모델 아키텍처

```text
베이스 모델  : MobileNetV2 (ImageNet 사전학습)
분류 계층   : Dropout(p=0.3) + Linear(1280→3)
학습 전략   : 2단계 전이학습
  Phase 1  : 백본 동결, Classifier만 학습 (3 에폭, lr=1e-3)
  Phase 2  : 백본 후반 5개 블록 해동 (12 에폭, lr=1e-5/1e-4)
손실 함수   : CrossEntropyLoss + WeightedRandomSampler
```

---

## 환경 설정

```bash
pip install -r requirements.txt
```

---

## 파일 역할

- `train_v3.py` : MobileNetV2 2단계 전이학습 코드
- `preprocess_affectnet.py` : AffectNet OpenCV 전처리
- `preprocess_mediapipe.py` : FER+ MediaPipe 전처리
- `merge_dataset.py` : 데이터 통합 및 클래스 균형
- `export_onnx.py` : PyTorch → ONNX 변환 및 자동 검증
