# BoN-TAPER 미팅 발표 아웃라인

## 1. 오늘 이야기할 핵심

이번 작업의 목표는 TAPER 논문의 핵심 아이디어를 그대로 가져오되, 내 연구 setting인 **BoN(best-of-N) parallel sampling**에 맞게 다시 정의해보는 것이다.

TAPER를 처음 보면 BoN과 비슷하게 보일 수 있다. 여러 candidate branch를 동시에 생성하고, deadline/SLO를 보면서 어느 branch를 더 진행할지 조절하기 때문이다. 하지만 실제 논문 setting은 BoN 전용이라기보다 여러 candidate universe를 동시에 전개하는 **multiverse / IRP류 setting**에 더 가깝다.

반면 vLLM의 BoN에서는 내부 scheduler 입장에서 각 branch가 거의 독립 request처럼 보인다. 그래서 TAPER를 그대로 적용하면 branch-level scheduling이 되고, 우리가 원하는 **logical BoN parent 단위의 completion objective**와 어긋날 수 있다.

그래서 이번 구현에서는 문제를 이렇게 다시 정의했다.

```text
BoN parent-level control problem:
  같은 logical request에 속한 N개 branch를 external_req_id로 묶고,
  parent별로 decode width를 조절한다.

mandatory work:
  active BoN parent마다 최소 1개 branch는 계속 progress시킨다.

opportunistic work:
  slack이 남을 때 같은 parent의 sibling branch를 추가로 admit한다.
```

현재 메인 구현은 다음 파일이다.

- `vllm/v1/core/sched/bon_taper_controller.py`
- policy name: `VLLM_TAPER_PLUS_POLICY=bon_taper`

## 2. Baseline과 비교 대상

### 2.1 IRP-EAGER / vLLM completion baseline

IRP-EAGER는 vLLM v1 completion baseline으로 보면 된다. 들어온 BoN branch들을 가능한 한 빨리 scheduler에 태운다.

장점:

- branch-level TTFT가 상대적으로 좋다.
- 많은 branch를 넓게 batch에 넣어서 raw token throughput이 높게 나온다.

단점:

- N이 크거나 output이 길면 너무 많은 branch가 동시에 살아남는다.
- KV/cache pressure와 queueing이 커진다.
- raw throughput은 높아도 SLO 안에 끝난 logical BoN request는 적을 수 있다.

### 2.2 Pure TAPER

기존 `scheduler.py`에 들어간 pure TAPER는 논문 재현에 가까운 경로다.

- policy name: `VLLM_TAPER_PLUS_POLICY=taper`
- branch들을 `external_req_id`로 묶기는 하지만, 기본적으로 branch-level TAPER logic을 scheduler에 붙인 형태다.
- parent당 하나의 baseline branch는 보호하지만, objective가 BoN parent completion에 완전히 특화된 것은 아니다.

발표에서 pure TAPER는 이렇게 설명하면 된다.

> TAPER 논문 아이디어를 vLLM scheduler에 먼저 붙여본 reproduction/ablation 경로다. 하지만 BoN에서는 user가 보는 단위가 branch가 아니라 logical parent이므로, pure TAPER만으로는 parent completion objective와 mismatch가 생길 수 있다.

### 2.3 BoN-TAPER

BoN-TAPER는 `bon_taper_controller.py`에서 구현한 새 경로다.

핵심 차이:

- 같은 `external_req_id`를 공유하는 branch들을 하나의 BoN parent로 묶는다.
- parent마다 최소 1개 branch를 mandatory progress로 보호한다.
- sibling branch는 slack budget 안에서만 추가 admit한다.
- 추가 branch는 `marginal parent utility / marginal predicted step cost`가 큰 순서로 greedy하게 선택한다.

즉 BoN-TAPER는 branch를 독립 request로만 보지 않고, **parent-level decode width allocation** 문제로 바꾼다.

## 3. 실험 세팅

현재 실험은 실제 QA dataset이 아니라 scheduling behavior를 보기 위한 synthetic workload다.

```text
model: Qwen/Qwen3.5-2B
prompt: synthetic prompt, 64 words
num_requests: 32 logical BoN parents
BoN N: 16, 32, 64
output_len: 1024, 2048
request_rate: 1 req/s
serial_fraction: 0.5
parallel_slo_mode: logical
repeat: 1
```

따라서 지금 결과는 모델 품질이나 accuracy가 아니라 scheduling 성능 결과다. 주요 지표는 다음이다.

| 지표 | 의미 |
|---|---|
| raw throughput | 전체 생성 token/s, `system_throughput_tokens_s` |
| user goodput | SLO를 만족한 logical user request 기준 useful token/s |
| SLO attainment | logical BoN parent가 SLO 안에 끝난 비율 |
| p99 TTFT | logical request가 첫 output token을 받을 때까지 걸린 시간의 p99 |

주의할 점:

- p99 TTFT는 BoN parent 완료 시간이 아니다.
- 현재 benchmark의 p99 TTFT는 logical request row 기준이다.
- parallel BoN request에서는 N개 branch 중 어떤 branch든 첫 output token을 낸 시점이 first token time으로 기록된다.
- 따라서 이 값은 "모든 sibling branch의 first token tail"이 아니라 "logical request가 처음 응답을 받기까지의 tail"에 가깝다.
- 하지만 TTFT 증가가 너무 크고 SLO 개선이 없으면 over-throttling으로 봐야 한다.

## 4. 메인 결과: IRP-EAGER vs BoN-TAPER

메인 결과는 다음 실험에서 나왔다.

- output folder: `benchmark_outputs/bon_taper_robust_profile_bon16_32_64_len1024_2048_rate1_req32_rep1_wsl`
- compact file: `comparison_compact_bon_taper.csv`

| N | out | EAGER goodput | BoN-TAPER goodput | SLO | EAGER raw | BoN-TAPER raw | raw ratio | p99 TTFT |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1024 | 0.000 | 10.527 | 0 -> 3.125% | 3580.86 | 2439.54 | 0.681x | 7.22 -> 26.05s |
| 16 | 2048 | 0.000 | 51.266 | 0 -> 18.75% | 3582.74 | 2269.43 | 0.633x | 20.40 -> 49.25s |
| 32 | 1024 | 0.000 | 0.000 | 0 -> 0% | 4193.19 | 2594.78 | 0.619x | 43.08 -> 99.74s |
| 32 | 2048 | 0.000 | 0.000 | 0 -> 0% | 4095.25 | 2225.30 | 0.543x | 80.23 -> 198.20s |
| 64 | 1024 | 0.000 | 0.000 | 0 -> 0% | 4269.57 | 2442.09 | 0.572x | 137.04 -> 283.67s |
| 64 | 2048 | 0.211 | 0.165 | 3.125 -> 3.125% | 4283.90 | 2229.78 | 0.521x | 219.94 -> 475.64s |

### 4.1 결과 해석

가장 긍정적인 케이스는 `N=16, output_len=2048`이다.

- EAGER는 SLO 0%, user goodput 0이었다.
- BoN-TAPER는 SLO를 18.75%까지 회복했고, user goodput도 51.266 tok/s로 올라갔다.
- 대신 raw throughput은 0.633x로 떨어졌고, p99 TTFT는 20.40s에서 49.25s로 증가했다.

즉 이 케이스에서는 BoN-TAPER가 raw throughput과 TTFT를 희생해서 logical SLO/goodput을 일부 살린다. 이게 우리가 기대한 방향이다.

`N=16, output_len=1024`에서도 약한 개선이 있다.

- SLO: 0 -> 3.125%
- goodput: 0 -> 10.527 tok/s
- raw throughput은 0.681x로 감소
- p99 TTFT는 7.22s에서 26.05s로 증가

하지만 `N=32/64`에서는 결과가 좋지 않다.

- raw throughput은 계속 0.52x~0.62x 수준으로 감소한다.
- p99 TTFT는 크게 증가한다.
- 그런데 SLO/goodput은 거의 회복하지 못한다.

이 구간은 단순히 "TAPER가 좋다"가 아니라, 현재 BoN-TAPER가 **too conservative / over-throttling**될 수 있음을 보여준다.

### 4.2 Core admission 강화 실험

startup lane 같은 별도 phase rule을 두지 않고, BoN-TAPER의 core admission 자체를 더 pressure-adaptive하게 바꾼 실험도 해봤다.

변경:

- low-pressure에서는 전체 running branch를 admit할 수 있게 full-admit fast path 추가
- hard feasibility reject를 완전히 쓰는 대신, TPOT 1 window 정도의 bounded soft budget 허용
- budget 초과분은 score penalty로 반영

결과:

| N | out | EAGER goodput | bounded BoN-TAPER goodput | SLO | raw ratio | p99 TTFT |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1024 | 223.320 | 216.714 | 3.125 -> 6.25% | 0.963x | 6.25 -> 9.35s |
| 16 | 2048 | 0.000 | 0.000 | 0 -> 0% | 0.905x | 18.48 -> 21.71s |
| 32 | 1024 | 0.000 | 0.000 | 0 -> 0% | 0.864x | 37.97 -> 53.62s |
| 32 | 2048 | 0.000 | 0.000 | 0 -> 0% | 0.914x | 70.78 -> 94.45s |

해석:

- 기존 보수적 BoN-TAPER보다 raw throughput과 TTFT는 크게 좋아졌다.
- `N=16, out=1024`에서는 SLO가 3.125%에서 6.25%로 올라갔다.
- 하지만 `N=16, out=2048`에서 기존 보수적 BoN-TAPER가 살렸던 SLO 회복은 사라졌다.
- 따라서 core admission을 너무 eager 쪽으로 열면 raw/TTFT는 좋아지지만, 긴 output에서 TAPER의 SLO recovery 효과가 약해질 수 있다.

현재 결론:

> 가장 단순한 bounded-soft core는 over-throttling을 줄이는 데는 효과가 있지만, 긴 output에서 completion SLO를 살리는 데는 부족하다. 다음 개선은 soft budget을 output length / parent completion urgency / memory pressure와 결합하는 방향이 더 좋아 보인다.

## 5. TTFT를 어떻게 해석할지

TTFT 표기 `7.22 -> 26.05s`는 다음 뜻이다.

```text
EAGER p99 logical TTFT = 7.22초
BoN-TAPER p99 logical TTFT = 26.05초
```

이 값은 BoN parent의 completion latency가 아니다. 현재 benchmark에서는 logical request가 첫 output token을 받기까지의 시간이다. parallel BoN request의 경우, N개 branch 중 어떤 branch든 처음 token을 내면 `first_token_time_s`가 찍힌다.

BoN-TAPER에서 TTFT가 증가하는 이유:

- EAGER는 가능한 branch를 빠르게 시작시킨다.
- BoN-TAPER는 parent별 width를 제한한다.
- 따라서 일부 logical request는 첫 branch가 prefill/decode에 들어가기까지 더 오래 기다릴 수 있다.
- 그 request-level queueing/admission delay가 p95/p99 TTFT에 잡힌다.

중요한 해석:

> TTFT 증가 자체는 admission control의 자연스러운 부작용일 수 있다. 하지만 TTFT가 크게 증가했는데 SLO/goodput 개선이 없다면, 그건 좋은 tradeoff가 아니라 over-throttling이다.

이번 결과에서는 `N=16` 일부 조건은 tradeoff가 어느 정도 성공했고, `N=32/64`는 실패한 조건으로 보는 게 맞다.

같은 결과를 p95 TTFT로 보면 다음과 같다.

| N | out | EAGER p95 TTFT | BoN-TAPER p95 TTFT | EAGER p99 TTFT | BoN-TAPER p99 TTFT |
|---:|---:|---:|---:|---:|---:|
| 16 | 1024 | 5.91s | 23.86s | 7.22s | 26.05s |
| 16 | 2048 | 19.11s | 41.28s | 20.40s | 49.25s |
| 32 | 1024 | 39.43s | 87.18s | 43.08s | 99.74s |
| 32 | 2048 | 73.19s | 173.98s | 80.23s | 198.20s |
| 64 | 1024 | 125.39s | 254.31s | 137.04s | 283.67s |
| 64 | 2048 | 200.04s | 438.28s | 219.94s | 475.64s |

즉 p99만 특이하게 튄 것이 아니라 p95도 크게 증가했다. 이건 tail 한두 개 문제가 아니라 admission/queueing 정책 전반이 더 보수적으로 동작했다는 신호다.

## 6. Pure TAPER와 BoN-TAPER의 차이

같은 setting에서 기존 `scheduler.py`의 pure TAPER도 `N=16,32`에 대해 돌려봤다.

- output folder: `benchmark_outputs/taper_scheduler_profile_bon16_32_len1024_2048_rate1_req32_rep1_wsl`
- compact file: `comparison_compact_taper.csv`

| N | out | EAGER goodput | Pure TAPER goodput | SLO | raw ratio | p99 TTFT |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1024 | 0.000 | 17.536 | 0 -> 9.375% | 0.719x | 8.95 -> 25.14s |
| 16 | 2048 | 0.000 | 0.000 | 0 -> 0% | 0.767x | 24.06 -> 19.71s |
| 32 | 1024 | 1.488 | 0.000 | 3.125 -> 0% | 0.686x | 41.62 -> 66.27s |
| 32 | 2048 | 0.000 | 0.000 | 0 -> 0% | 0.798x | 86.13 -> 114.49s |

해석:

- Pure TAPER는 `N=16, out=1024`에서는 BoN-TAPER보다 더 좋았다.
- 하지만 `N=16, out=2048`에서는 BoN-TAPER만 SLO/goodput을 살렸다.
- `N=32`에서는 pure TAPER가 EAGER보다 나빠지는 조건도 있었다.

따라서 발표에서는 이렇게 말하는 게 안전하다.

> Pure TAPER는 branch-level objective에 더 가까워 일부 조건에서는 잘 작동하지만, BoN parent completion objective와 항상 맞지는 않는다. BoN-TAPER는 objective는 더 맞지만, 현재 구현은 아직 보수적이라 raw throughput과 TTFT penalty가 크다.

## 7. 현재 결론

현재 결과를 한 문장으로 요약하면 다음과 같다.

> BoN-TAPER는 BoN parent-level objective에 맞게 TAPER를 재정의한 구현이고, `N=16`의 일부 pressure 조건에서는 EAGER가 놓치는 logical SLO/goodput을 회복한다. 하지만 `N=32/64`에서는 raw throughput과 TTFT를 크게 희생하면서도 SLO를 거의 회복하지 못해, 현재 정책은 아직 over-throttling 문제가 있다.

조금 더 발표용으로 부드럽게 말하면:

> 지금 단계에서는 BoN-TAPER가 최종적으로 우월하다는 주장보다는, TAPER를 BoN에 그대로 붙이면 objective mismatch가 생기고, parent-level control로 바꾸면 일부 조건에서 가능성이 보인다는 것을 확인했다. 다음 단계는 raw throughput/TTFT penalty를 줄이면서 SLO recovery를 유지하는 것이다.

## 8. TAPER+ 방향

BoN-TAPER의 현재 병목은 단순 width control만으로는 어떤 branch를 먼저 살릴지 충분히 잘 고르지 못한다는 점이다.

특히 BoN에서는 다음 정보가 중요하다.

- 어떤 branch가 곧 끝날 것인가?
- 어떤 branch가 KV memory를 오래 잡아먹을 것인가?
- 지금 GPU KV cache free block이 충분한가?
- sibling branch를 늦추면 parent completion deadline에 얼마나 영향을 주는가?

그래서 TAPER+는 BoN-TAPER 위에 **memory/lifetime awareness**를 추가하는 방향이다.

현재 계획:

```text
bon_taper:
  parent deadline/slack 기반 width control

bon_taper_plus:
  bon_taper
  + branch short-lifetime probability
  + future KV footprint estimate
  + free KV block / memory pressure gate
```

구현 아이디어:

- `prob_short`가 높은 branch를 먼저 admit해서 빨리 끝내고 KV를 회수한다.
- future footprint가 큰 branch는 memory pressure가 있을 때 defer한다.
- 단, parent당 baseline branch는 계속 보호한다.
- overdue sibling branch는 완전히 굶지 않도록 boost를 준다.

발표용 설명:

> Pure BoN-TAPER는 deadline-aware width control만 한다. TAPER+에서는 여기에 branch lifetime과 KV memory pressure를 넣어서, 곧 끝날 branch는 먼저 끝내고 long-lived branch는 pressure 상황에서 조절하려고 한다. 목표는 N이 커질 때 raw throughput과 TTFT penalty를 줄이면서 logical SLO/goodput을 회복하는 것이다.

## 9. 교수님께 말할 포인트

### 시작 설명

> TAPER를 BoN setting에 맞춰 재현해보고 있습니다. 다만 TAPER 논문 setting은 완전한 BoN이라기보다 multiverse/IRP setting에 가까워서, vLLM BoN에서는 branch들을 external_req_id로 다시 묶고 parent-level width control 문제로 재정의했습니다.

### 구현 설명

> 기존 vLLM EAGER는 BoN branch를 가능한 한 많이 바로 실행합니다. 반면 BoN-TAPER는 parent마다 최소 1개 branch를 보호하고, sibling branch는 slack과 predicted step cost를 보고 추가 admit합니다.

### 결과 설명

> `N=16`에서는 일부 조건에서 EAGER가 SLO를 전혀 만족하지 못할 때 BoN-TAPER가 logical SLO/goodput을 일부 회복했습니다. 특히 output length 2048에서는 SLO가 0%에서 18.75%로 올라갔습니다. 하지만 raw throughput은 낮아지고 logical request의 p95/p99 TTFT는 증가했습니다.

### 한계 설명

> `N=32/64`에서는 현재 BoN-TAPER가 raw throughput과 TTFT를 희생하면서도 SLO를 거의 회복하지 못했습니다. 이는 단순 parent-level throttling만으로는 충분하지 않고, branch lifetime이나 KV memory pressure를 같이 봐야 한다는 신호로 보고 있습니다.

### 다음 단계

> 다음은 TAPER+ 방향입니다. branch가 곧 끝날 가능성, future KV footprint, free KV blocks를 admission에 넣어서, memory pressure 상황에서는 long-lived branch를 조절하고 short-lived branch를 먼저 끝내는 방식으로 개선하려고 합니다.

## 10. 예상 질문과 답

### Q. BoN-TAPER가 EAGER보다 항상 좋은가?

아니다. 현재 결과에서는 `N=16` 일부 조건에서만 logical SLO/goodput 개선이 보인다. `N=32/64`에서는 아직 좋지 않다. 그래서 최종 성능 주장보다는 problem reformulation과 다음 개선점 제시가 핵심이다.

### Q. TTFT가 너무 늘어나는데 괜찮은가?

완전히 괜찮다고 보기는 어렵다. admission control 때문에 logical TTFT가 늘어나는 것은 어느 정도 예상되지만, SLO 개선 없이 TTFT만 늘어나면 over-throttling이다. 현재 큰 N 조건에서는 이 문제가 보인다.

### Q. parent마다 최소 1개 branch 보장은 이미 있는가?

있다. 현재 구현은 running decode candidate 안에서 parent당 하나의 baseline branch를 보호한다. 다만 이것이 모든 sibling branch의 TTFT cap을 보장하지는 않는다. deferred sibling branch가 오래 밀리면 p99 TTFT는 여전히 커질 수 있다.

### Q. 왜 TAPER+가 필요한가?

BoN에서는 어떤 branch를 더 진행할지가 중요하다. 단순 slack만 보면 곧 끝날 branch와 오래 살아남을 branch를 구분하지 못한다. TAPER+는 branch lifetime과 KV memory footprint를 보고 더 똑똑하게 admit하려는 방향이다.
