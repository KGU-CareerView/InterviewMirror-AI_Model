# InterviewMirror-AI Model Spec

## 모델 개요

AffectNet + FER+ 데이터셋을 기반으로 학습한 MobileNetV2 면접 표정 감정 분류 모델입니다.
면접 상황에 특화된 3개 클래스(Stable / Nervous / Neutral)로 분류합니다.

---

## 클래스 정의

| ID | 클래스 | 의미 |
|---|---|---|
| 0 | Stable | 안정적이고 자신감 있는 상태 |
| 1 | Nervous | 긴장하거나 불안한 상태 |
| 2 | Neutral | 감정 표현이 없는 중립 상태 |

---

## 입력 스펙

```text
텐서 형태 : [1, 3, 224, 224]
색 공간   : RGB (OpenCV 사용 시 BGR → RGB 변환 필수)
스케일    : /255.0
mean      : [0.485, 0.456, 0.406]
std       : [0.229, 0.224, 0.225]
```

---

## 출력 스펙

```text
텐서 형태 : [1, 3]
후처리    : softmax → argmax → 클래스 ID
```

---

## 전처리 코드

```python
import cv2
import numpy as np

def preprocess(frame, face_bbox):
    x, y, w, h = face_bbox

    # 얼굴 크롭
    face = frame[y:y+h, x:x+w]

    # BGR → RGB 변환 (필수)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # 224×224 리사이즈
    face = cv2.resize(face, (224, 224))

    # 정규화
    face = face.astype(np.float32) / 255.0
    face = (face - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    # HWC → CHW + 배치 추가
    face = np.transpose(face, (2, 0, 1))
    face = np.expand_dims(face, axis=0)

    return face.astype(np.float32)
```

---

## 성능 (v3)

```text
Accuracy      : 89.6%
Macro F1      : 0.899
Nervous Recall: 97.1%
CPU 추론 속도  : 28ms (ONNX Runtime 기준)
```

---

## ONNX 변환 정보

```text
Opset 버전  : 13
입력 이름   : input
출력 이름   : output
정밀도      : FP32
최대 오차   : < 1e-4 (PyTorch ↔ ONNX Runtime)
```

---

## 모델 다운로드

| 파일 | 용도 |
|---|---|
| `interview_best_model_v3.pth` | PyTorch 추가 학습용 |
| `interview_model_v3.onnx` | ONNX Runtime 추론용 |
| `interview_model_v3.onnx.data` | onnx와 반드시 같은 폴더에 위치 |


---

## 파일 역할

- `train_v3.py` : MobileNetV2 2단계 전이학습 코드
- `preprocess_affectnet.py` : AffectNet OpenCV 전처리
- `preprocess_mediapipe.py` : FER+ MediaPipe 전처리
- `merge_dataset.py` : 데이터 통합 및 클래스 균형
- `export_onnx.py` : PyTorch → ONNX 변환 및 자동 검증