---
title: 8장 2강 Argument Validation과 함수 실행
date: 2026-09-13
updated: 2026-09-13
description: KANT 강의 '8장 2강 Argument Validation과 함수 실행' 정리
---

## 1. Argument 검증이 필요한 이유

Strict Schema = 출력의 구조와 자료형을 엄격히 맞추는 것

모델이 strict Schema를 따르더라도 애플리케이션의 실행 경계에서는 다시 검증해야 한다


arguments
```json
{
  "order_id": "../../admin/secrets",
  "include_history": true
}
```

자료형은 문자열과 boolean이라 Schema를 만족할 수 있지만, `order_id` 값이 실제 주문 ID 형식은 아니다.

```json
{
  "name": "delete_all_orders",
  "arguments": {}
}
```

