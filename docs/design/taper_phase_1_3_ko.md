# 순수 TAPER 구현 정리

이 문서는 현재 로컬 vLLM v1 scheduler에 들어간 **순수 TAPER** 구현을 정리한다. 같은 코드 경로 안에 TAPER+도 함께 들어가 있지만, 순수 TAPER는 `VLLM_TAPER_PLUS_POLICY=taper`로 선택되는 ablation 모드이며, TAPER+의 메모리 게이트나 branch lifetime 기반 정렬을 사용하지 않는다.

구현 위치는 주로 다음 파일이다.

- `vllm/v1/core/sched/scheduler.py`
- `vllm/envs.py`
- `vllm/v1/request.py`
- `vllm/v1/core/sched/output.py`
- `benchmarks/benchmark_taper_plus.py`

## 1. 실행 모드

순수 TAPER는 별도 scheduler 클래스를 새로 만든 것이 아니라, TAPER+와 공통으로 추가한 scheduler hook 안에서 policy 값으로 분기한다.

```bash
VLLM_ENABLE_TAPER_PLUS=1
VLLM_TAPER_PLUS_POLICY=taper
```

이름은 `TAPER_PLUS` 환경 변수 prefix를 공유하지만, policy가 `taper`이면 다음 기능만 켜진다.

- BoN child branch를 logical parent request 단위로 묶기
- 각 parent group마다 최소 1개 branch baseline 보호
- TTFT/TPOT SLO로 next-token deadline 계산
- deadline slack에 따라 이번 step에서 추가 branch를 얼마나 더 admit할지 결정
- latency model을 사용해 decode width 증가 비용 예측
- greedy utility-per-cost 방식으로 추가 branch 선택

반대로 policy가 `taper`일 때는 다음 TAPER+ 전용 기능을 쓰지 않는다.

- `prob_short`가 높은 branch 우선 정렬
- KV cache free block 기반 memory pressure gate
- branch future footprint 기반 admission 제한
- short-lived branch만 memory pressure에서 통과시키는 조건

## 2. 왜 group 단위로 보는가

vLLM v1은 `SamplingParams(n > 1)` 요청을 여러 child `Request`로 나누어 처리한다. 예를 들어 `n=32` 요청 하나는 scheduler 내부에서 32개의 독립 branch처럼 보인다.

순수 TAPER는 이 child branch들을 완전히 독립된 요청으로 다루지 않고, 같은 사용자 요청에서 나온 branch들을 하나의 logical request group으로 묶는다. 이때 사용하는 key는 다음 helper에서 정한다.

```python
def _taper_plus_group_id(request: Request) -> str:
    return request.external_req_id or request.request_id
```

BoN child들은 같은 `external_req_id`를 공유하므로 하나의 group으로 묶인다. 일반 단일 요청은 group 크기가 1이라 width regulation 대상이 되지 않는다.

## 3. Decode candidate만 조절한다

순수 TAPER는 prefill 단계가 아니라, 이미 decode 단계에 들어간 running request의 branch 폭을 조절한다. candidate 판정은 다음 조건으로 한다.

```python
request.sampling_params is not None
and request.num_output_tokens + request.num_output_placeholders > 0
and request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
    > 0
```

즉, 출력 token이 이미 있거나 async placeholder가 있고, 아직 이번 step에서 계산할 token이 남아 있는 branch만 TAPER admission 대상이다. 초기 prefill, chunked prefill, prefix cache, encoder cache, speculative decoding의 기본 흐름은 기존 scheduler 경로를 그대로 탄다.

## 4. Admission planner 흐름

핵심 함수는 `Scheduler._get_taper_plus_admitted_running_req_ids(token_budget)`이다. policy 이름은 TAPER+를 포함하지만, 내부에서 `VLLM_TAPER_PLUS_POLICY=taper`이면 순수 TAPER 로직만 수행한다.

### 4.1 Regulated group 찾기

먼저 `self.running`에 있는 request 중 decode candidate만 모아 group을 만든다.

```python
group_to_candidates[group_id].append(request)
```

그 다음 group 내 candidate가 2개 이상인 group만 regulation 대상으로 본다.

```python
regulated_groups = {
    group_id
    for group_id, requests in group_to_candidates.items()
    if len(requests) > 1
}
```

regulated group이 없으면 `None`을 반환하고, scheduler는 기존 vLLM처럼 모든 running request를 순서대로 처리한다.

### 4.2 Baseline protection

각 active BoN parent group마다 첫 번째 decode branch 하나는 무조건 admit한다.

```python
if group_id not in baseline_groups:
    admitted_req_ids.add(request.request_id)
    baseline_groups.add(group_id)
```

이 부분이 TAPER의 안전장치다. branch 폭을 줄이더라도 어떤 parent request가 완전히 굶지 않도록 최소 1개 branch의 next-token 진행은 보호한다.

regulated group에 속하지 않거나 decode candidate가 아닌 request는 TAPER가 건드리지 않고 admit set에 그대로 포함한다.

### 4.3 Deferred branch 준비

baseline으로 보호된 branch 외의 sibling branch들은 `deferred_by_group`에 모아 둔다.

순수 TAPER에서는 deferred branch를 다음 순서로 정렬한다.

```python
deferred.sort(
    key=lambda request: (request.arrival_time, request.request_id)
)
```

즉 FIFO에 가까운 순서다. TAPER+와 달리 `prob_short`를 보지 않는다. 이 점이 순수 TAPER와 TAPER+의 가장 중요한 차이 중 하나다.

## 5. Deadline과 slack budget

순수 TAPER는 고정 branch 개수를 쓰지 않고, 현재 batch에서 가장 급한 logical request의 slack을 보고 추가 branch 폭을 정한다.

### 5.1 Request group deadline

deadline은 child branch 단위가 아니라 `external_req_id` group 단위로 계산한다.

아직 group의 어떤 branch도 output token을 내지 않았다면 TTFT SLO를 쓴다.

```python
deadline = min(child.arrival_time for child in group) + TTFT_SLO
```

이미 output token이 하나라도 나온 group이면 TPOT SLO를 쓴다. 이때 group 내에서 가장 최근에 token을 낸 branch의 시간을 기준으로 한다.

```python
deadline = max(child.taper_last_token_time for child in group) + TPOT_SLO
```

이를 위해 `Request`에는 `taper_last_token_time`이 추가되어 있고, `append_output_token_ids()`에서 새 output token이 붙을 때마다 갱신된다.

### 5.2 Step budget

baseline branch들만 실행했을 때의 예상 latency를 먼저 계산한다.

```python
baseline_step_ms = self._predict_taper_plus_step_ms(
    active_decode_width, active_context_tokens
)
```

그 다음 active decode group들 중 가장 작은 slack을 구한다.

```python
min_slack_ms = min((group_deadline - now) * 1000.0)
```

최종 budget은 다음 형태다.

```python
budget_ms = baseline_step_ms + rho * max(0, min_slack_ms - baseline_step_ms)
```

`rho`는 `VLLM_TAPER_PLUS_RHO`로 조절한다. `rho=1.0`이면 slack이 허용하는 만큼 IRP-EAGER에 가까워질 수 있고, `rho=0.0`이면 baseline 보호 폭에 더 가깝게 동작한다.

현재 구현은 여기에 두 가지 보정을 추가한다.

첫째, global `min_slack`만 쓰면 가장 급한 request 하나가 모든 BoN group의 추가 branch admission을 막을 수 있다. 그래서 각 후보 branch를 평가할 때는 global budget과 해당 parent group의 own-slack budget 중 더 큰 값을 사용한다.

```python
candidate_budget_ms = max(global_budget_ms, group_budget_ms)
```

둘째, logical parent deadline은 group 안의 어느 branch 하나가 계속 token을 내면 갱신되므로, 다른 sibling branch가 오래 기다리는 상황을 숨길 수 있다. 이를 막기 위해 deferred branch마다 branch-level deadline 초과 시간을 계산하고, 오래 굶은 branch는 budget 검사에서 바로 탈락시키지 않고 greedy 경쟁에 참여시킨다.

```python
branch_overdue_ms = now - branch_deadline
```

이 값은 score에도 곱해져서, 오래 기다린 sibling branch가 다음 admission에서 더 높은 우선순위를 갖는다. 이 보정의 목적은 eager fallback이 아니라, slack이 충분하거나 sibling tail이 커지는 상황에서 TAPER planner 자체가 자연스럽게 더 많은 branch를 admit하도록 만드는 것이다.

## 6. Latency model

순수 TAPER는 추가 branch를 admit했을 때 이번 scheduler step의 시간이 얼마나 늘어날지 예측한다.

현재 model은 decode width와 context length를 사용하는 단순 선형식이다.

```python
predicted_ms =
    base_ms
    + width_ms * decode_width
    + context_ms_per_1k * (context_tokens / 1000)
```

예측값에는 `VLLM_TAPER_PLUS_SAFETY_FACTOR`가 곱해진다. 기본값은 1.10이다.

초기 `width_ms`는 `VLLM_TAPER_PLUS_LATENCY_PER_SEQ_MS`에서 온다. 또한 `VLLM_TAPER_PLUS_ONLINE_CALIBRATION=1`이면 실제 step 시간이 관측될 때마다 model coefficient를 조금씩 갱신한다.

관측은 `schedule()`이 `SchedulerOutput`을 만들기 직전에 시작 시간을 기록하고, `update_from_output()`에서 실제 model runner 결과를 받은 뒤 elapsed time을 계산하는 방식이다.

단, prefill token이 섞인 step은 decode width cost와 성격이 다르기 때문에 online calibration에서 제외한다.

## 7. Greedy utility-per-cost 선택

baseline 이후 남은 token budget이 있으면, 순수 TAPER는 deferred branch 중 어떤 branch를 추가로 admit할지 greedy하게 고른다.

각 반복에서 group마다 다음 candidate 하나를 보며, 그 branch를 추가했을 때 예상 step latency가 `budget_ms`를 넘는지 확인한다.

```python
predicted_step_ms = self._predict_taper_plus_step_ms(
    active_decode_width + 1,
    active_context_tokens + request.num_computed_tokens,
)

if predicted_step_ms > budget_ms:
    infeasible_groups.add(group_id)
    continue
```

budget 안에 들어오면 marginal utility 대비 marginal latency cost 점수를 계산한다.

```python
score = du / (eps + max(0.0, predicted_step_ms - baseline_step_ms))
```

가장 높은 score를 가진 branch를 admit하고, active decode width/context를 갱신한 뒤 같은 과정을 반복한다.

utility는 기본적으로 linear이다.

```python
def _taper_plus_marginal_utility(self, granted: int) -> float:
    if self.taper_plus_utility == "concave":
        return (granted + 1) ** 0.5 - granted**0.5
    return 1.0
```

`linear`는 throughput 중심 TAPER 실험에 맞고, `concave`는 group 간 fairness 성향을 보는 ablation으로 볼 수 있다.

## 8. 실제 schedule()에 연결되는 방식

`schedule()`에서는 KV cache step 초기화 직후 admit set을 계산한다.

```python
self.kv_cache_manager.new_step_starts()
taper_plus_admitted_running_req_ids = (
    self._get_taper_plus_admitted_running_req_ids(token_budget)
)
```

그리고 running request를 순회할 때 admit set에 없는 request는 이번 forward pass에서 건너뛴다.

```python
if (
    taper_plus_admitted_running_req_ids is not None
    and request.request_id not in taper_plus_admitted_running_req_ids
):
    req_index += 1
    continue
```

중요한 점은 skip된 branch를 preempt하지 않는다는 것이다. KV cache를 강제로 해제하지도 않는다. 단지 이번 scheduler step의 `scheduled_running_reqs`에 포함하지 않아서 forward pass에 들어가지 않게 할 뿐이다. 다음 step에서는 다시 candidate가 될 수 있다.

따라서 기존 vLLM의 allocation, preemption, LoRA, encoder cache, speculative decoding, scheduler output 생성 경로는 admit된 request에 대해 그대로 유지된다.

## 9. 순수 TAPER와 TAPER+의 차이

| 항목 | 순수 TAPER (`policy=taper`) | TAPER+ (`policy=taper_plus`) |
|---|---|---|
| Parent group baseline 보호 | 사용 | 사용 |
| TTFT/TPOT deadline 기반 slack budget | 사용 | 사용 |
| Greedy utility-per-cost branch 선택 | 사용 | 사용 |
| Deferred branch 정렬 | FIFO 성격: `arrival_time`, `request_id` | `prob_short` 높은 branch 우선 |
| KV free block threshold | 사용 안 함 | 사용 |
| Future footprint estimate | 사용 안 함 | 사용 |
| Memory pressure에서 short-lived branch만 admit | 사용 안 함 | 사용 |

정리하면, 순수 TAPER는 “현재 co-batched logical request들의 deadline slack이 얼마나 남았는가”를 기준으로 BoN branch 병렬 폭을 조절한다. 메모리 회수 가능성이나 branch가 곧 끝날 확률은 고려하지 않는다. 그래서 scheduler.py 안의 순수 TAPER 구현은 TAPER+의 메모리-aware 정책을 빼고, 논문식 IRP width controller만 분리해서 재현하는 역할을 한다.

## 10. Benchmark에서의 사용

`benchmarks/benchmark_taper_plus.py`에서는 mode로 순수 TAPER를 실행할 수 있다.

```bash
python benchmarks/benchmark_taper_plus.py --mode taper
```

이 mode는 내부적으로 다음 환경 변수를 설정한다.

```python
VLLM_ENABLE_TAPER_PLUS = "1"
VLLM_TAPER_PLUS_POLICY = "taper"
```

비교 실험에서는 다음 mode들이 함께 쓰인다.

- `irp_eager`: stock eager BoN scheduling에 가까운 비교군
- `taper`: 순수 TAPER
- `taper_plus`: TAPER + lifetime/memory-aware 확장
- `all`: IRP-OFF, IRP-EAGER, TAPER, TAPER+ 전체 비교
