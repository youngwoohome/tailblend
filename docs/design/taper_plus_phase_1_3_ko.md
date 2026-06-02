# TAPER+ Phase 1-3 한국어 설명

이 문서는 현재 vLLM v1 스케줄러 구조 위에 추가한 TAPER+ 변경 사항을
한국어로 정리한다. 핵심 목표는 `SamplingParams(n > 1)`처럼 하나의 사용자
요청이 여러 개의 BoN(best-of-N) 브랜치로 갈라지는 상황에서, 모든
브랜치를 항상 동시에 돌리는 대신 각 부모 요청 안의 디코드 폭을 상황에
따라 조절하는 것이다.

기본 vLLM은 `n=32` 요청을 받으면 내부적으로 32개의 child `Request`로
나누고, 스케줄러는 이 child들을 서로 독립적인 요청처럼 다룬다. TAPER+는
여기에 세 가지 정보를 추가한다.

- 각 브랜치가 곧 끝날 가능성
- 현재 KV cache 여유 블록과 앞으로 필요한 메모리 양
- 이번 스텝에 몇 개의 디코드 브랜치를 실제 forward pass에 넣을지 결정하는
  admission policy

## Phase 1: 브랜치 수명 예측기

Phase 1은 각 `Request`에 `prob_short`라는 값을 추가한다. 이 값은 해당
브랜치가 짧은 시간 안에 끝날 가능성을 의미한다. 초기값은 `0.0`이고,
브랜치가 짧게 끝날 것 같다고 판단되면 값이 올라가며 다시 낮아지지는 않는다.

현재 구현은 두 가지 신호를 사용한다.

- 마지막 생성 토큰이 EOS 또는 stop token이면 `prob_short = 1.0`으로 둔다.
- 그렇지 않으면 지금까지 생성한 토큰 수가 `max_tokens`에 얼마나 가까운지로
  진행도를 계산한다.

즉, 이미 출력 길이 제한에 가까워진 브랜치나 종료 토큰을 낸 브랜치는 곧
끝날 가능성이 높은 브랜치로 취급된다. 이 정보는 Phase 3에서 어떤 브랜치를
먼저 실행할지 정렬할 때 사용된다.

또한 `Request`에 `external_req_id`를 보관한다. BoN 요청에서는 child
request마다 내부 `request_id`는 다르지만, 사용자 관점의 원래 요청 ID는
같다. `external_req_id`는 같은 BoN 부모에서 나온 child 브랜치들을 하나의
그룹으로 묶기 위해 필요하다.

## Phase 2: 메모리 안전 게이트

Phase 2는 스케줄러가 KV cache 압박을 볼 수 있도록 `KVCacheManager`에 작은
helper를 추가한다.

`get_remaining_free_blocks()`는 현재 남아 있는 GPU KV block 수를 반환한다.
스케줄러가 `BlockPool` 내부 구현을 직접 건드리지 않고도 메모리 여유를
확인할 수 있게 하기 위한 인터페이스다.

`estimate_future_footprint(request)`는 특정 브랜치가 앞으로 완료될 때까지
추가로 필요할 수 있는 KV block 수를 추정한다.

- 긴 브랜치로 보이면 `prompt tokens + max_tokens`까지 갈 수 있다고 보고
  보수적으로 계산한다.
- 짧은 브랜치로 보이면 `prob_short`를 반영해 남은 출력 토큰 수를 더 작게
  잡는다.

실제 block 수 계산은 기존 vLLM의 KV coordinator에 맡긴다. 이렇게 하면
block size, hybrid KV cache, sliding window, admission cap 같은 기존 vLLM
정책과 추정 로직이 어긋나지 않는다.

## Phase 3: 이중 예산 기반 플래너

Phase 3은 실제 스케줄러 admission policy를 추가한다. 이 기능은 환경 변수로
opt-in되며, `VLLM_ENABLE_TAPER_PLUS=0`일 때는 stock vLLM과 같은 경로로
동작한다.

주요 환경 변수는 다음과 같다.

- `VLLM_ENABLE_TAPER_PLUS`: TAPER+ 스케줄링 활성화 여부
- `VLLM_TAPER_PLUS_MEMORY_THRESHOLD_BLOCKS`: 메모리 압박으로 볼 free block
  기준값
- `VLLM_TAPER_PLUS_STEP_TARGET_MS`: 한 decode step의 목표 시간 예산
- `VLLM_TAPER_PLUS_LATENCY_PER_SEQ_MS`: 브랜치 하나당 latency proxy

스케줄러는 먼저 같은 `external_req_id`를 가진 child request들을 하나의 BoN
그룹으로 묶는다. 일반 단일 요청은 사실상 자기 자신만의 그룹이 된다.

그 다음 현재 running request 중에서 이번 step에 실행할 브랜치를 고른다.
정책은 다음 순서로 동작한다.

1. 각 활성 BoN 그룹에서 최소 하나의 브랜치는 admit한다. 이렇게 해야 어떤
   부모 요청도 완전히 굶지 않고 계속 조금씩 진행할 수 있다.
2. 남은 브랜치들은 `prob_short`가 높은 순서로 우선 시도한다. 곧 끝날
   가능성이 높은 브랜치를 먼저 실행하면 KV cache를 빨리 반환할 가능성이
   커진다.
3. 브랜치를 하나 더 넣었을 때 예측 step latency가
   `VLLM_TAPER_PLUS_STEP_TARGET_MS`를 넘으면 admit하지 않는다.
4. free KV block이 threshold 아래이거나, 해당 브랜치의 미래 footprint가
   가상 여유 block보다 크면 메모리 압박 상태로 본다. 이때는
   `is_short_lived()`인 브랜치만 추가 admit한다.

중요한 점은 admit되지 않은 브랜치를 preempt하거나 KV cache를 해제하지
않는다는 것이다. 그 브랜치는 이번 forward pass에서만 빠지고, 다음
스케줄링 step에서 다시 후보가 된다. 따라서 기존 vLLM의 allocation,
preemption, LoRA, encoder cache, speculative decoding 경로는 admit된
브랜치에 대해 그대로 유지된다.

요약하면 Phase 1-3의 TAPER+는 BoN child 브랜치들을 부모 요청 단위로 묶고,
곧 끝날 브랜치를 우선 실행하며, 시간 예산과 KV cache 예산을 동시에 보면서
디코드 폭을 조절하는 스케줄러 확장이다. 기본 동작은 환경 변수로 꺼져 있기
때문에 stock vLLM 호환성을 유지한다.
