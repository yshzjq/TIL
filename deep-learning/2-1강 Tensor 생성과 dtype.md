---
title: 2-1강 Tensor 생성과 dtype
date: 2026-08-19
updated: 2026-08-19
description: KANT 강의 '2-1강 Tensor 생성과 dtype' 정리
---

## 1. 이번에 우리가 배울 것

딥러닝 모델은 숫자를 입력받고 숫자를 출력

이미지도 숫자, 텍스트도 숫자, 정답 라벨도 숫자로 바꿔서 모델에 넣는다

이때 PyTorch에서 숫자 데이터를 담는 기본 그릇이 **Tensor**다.

Tensor는 다음과 같이 생각하면 됩니다.

| 비유 | 딥러닝에서의 의미 |
| --- | --- |
| 엑셀 표 | 2차원 Tensor |
| 여러 장의 엑셀 표 | 3차원 Tensor |
| 여러 장의 컬러 이미지 묶음 | 4차원 Tensor |
| 모델이 읽는 숫자 상자 | Tensor |

딥러닝에서 Tensor를 볼 때는 값 자체보다 먼저 **shape, dtype, device**를 확인해야 한다

<img src="{{ '/assets/images/uploads/deep-learning/Tensor차원구조.png' | relative_url }}" alt="Tensor차원구조.png" loading="lazy">

이미지 설명: Scalar, Vector, Matrix, Tensor를 한 번에 비교하는 그림

## 2. Tensor의 차원 이해하기

Tensor의 차원은 “몇 겹으로 숫자가 감싸져 있는가”를 뜻합니다.

### 0차원 Tensor: Scalar

숫자 하나입니다.

```python
import torch

# 숫자 하나를 Tensor로 만듭니다.
# 이런 Tensor를 scalar라고 부릅니다.
scalar = torch.tensor(7)

print(scalar)
print(scalar.shape)
print(scalar.ndim)
```

출력 결과

```
tensor(7)
torch.Size([])
0
```

`torch.Size([])`는 비어 있는 것처럼 보이지만, 오류가 아닙니다.

값이 하나뿐이라서 행도 열도 없는 **0차원 Tensor**라는 뜻이다.






