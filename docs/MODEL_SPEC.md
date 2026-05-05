\# 모델 명세 (MODEL\_SPEC)



\## 모델 다운로드

| 파일 | 용도 | 링크 |

|---|---|---|

| interview\_best\_model\_v3.pth | PyTorch 추가 학습용 | \[Google Drive](링크) |

| interview\_model\_v3.onnx + .onnx.data | ONNX Runtime 추론용 | \[Google Drive](링크) |



> .onnx와 .onnx.data는 반드시 같은 폴더에 있어야 합니다.



\## 입력 스펙

| 항목 | 값 |

|---|---|

| 형태 | \[1, 3, 224, 224] |

| 채널 | RGB (OpenCV BGR→RGB 변환 필수) |

| mean | \[0.485, 0.456, 0.406] |

| std | \[0.229, 0.224, 0.225] |



\## 출력 스펙

| 인덱스 | 클래스 |

|---|---|

| 0 | Stable (안정) |

| 1 | Nervous (긴장) |

| 2 | Neutral (무표정) |



\## 성능 (v3)

| 지표 | 값 |

|---|---|

| Accuracy | 89.6% |

| Macro F1 | 0.899 |

| Nervous Recall | 97.1% |

| CPU 추론 속도 | 28ms |



\## 전처리 코드

```python

import cv2, numpy as np



def preprocess(frame, face\_bbox):

&#x20;   x, y, w, h = face\_bbox

&#x20;   face = frame\[y:y+h, x:x+w]

&#x20;   face = cv2.cvtColor(face, cv2.COLOR\_BGR2RGB)

&#x20;   face = cv2.resize(face, (224, 224))

&#x20;   face = face.astype(np.float32) / 255.0

&#x20;   face = (face - \[0.485,0.456,0.406]) / \[0.229,0.224,0.225]

&#x20;   face = np.transpose(face, (2,0,1))

&#x20;   return np.expand\_dims(face, axis=0).astype(np.float32)



