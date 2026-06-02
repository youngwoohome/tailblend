# TAPER / TAPER+ WSL 실험 실행 가이드

이 문서는 다른 에이전트가 같은 Windows + WSL 환경에서 TAPER 실험을
재현할 수 있도록 남기는 실행 절차이다. 코드 설명이 아니라 **실험을
어떻게 돌리고, 결과를 어디서 보고, 어떤 점을 조심해서 해석해야 하는지**
에 초점을 둔다.

## 기본 환경

현재 repo 위치:

```text
C:\Users\young\OneDrive\바탕 화면\vllm-src
```

WSL에서 접근하는 경로:

```bash
/mnt/c/Users/young/OneDrive/바탕\ 화면/vllm-src
```

사용한 WSL distro:

```bash
Ubuntu-24.04
```

사용한 Python 가상환경:

```bash
~/.venvs/vllm-taper
```

대부분의 실험은 PowerShell Python이 아니라 WSL venv에서 실행해야 한다.
PowerShell 쪽 Python에는 CUDA/Torch/vLLM 환경이 맞지 않을 수 있다.

기본 실행 prefix:

```powershell
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/c/Users/young/OneDrive/바탕\ 화면/vllm-src && source ~/.venvs/vllm-taper/bin/activate && <COMMAND>"
```

FlashInfer sampler JIT 문제를 피하기 위해 실험 명령 앞에 보통 아래 환경
변수를 둔다.

```bash
VLLM_USE_FLASHINFER_SAMPLER=0
```

## 주요 스크립트

단일 benchmark:

```text
benchmarks/benchmark_taper_plus.py
```

Grid sweep runner:

```text
benchmarks/run_taper_plus_sweep.py
```

이름은 `taper_plus`지만 현재 pure TAPER 실험도 같은 스크립트에서 돌린다.

주요 mode:

```text
irp_eager  : stock vLLM 방식, n개 branch를 eager하게 실행
taper      : 논문 Algorithm 1 기반 pure TAPER
taper_plus : TAPER + memory/lifetime gate
both       : irp_eager vs taper
both_plus  : irp_eager vs taper_plus
all        : irp_off, irp_eager, taper, taper_plus
```

현재 pure TAPER 검증에는 보통 `--mode both` 또는 sweep runner의
`--compare-mode both`를 사용한다.

## 공통 실험 옵션

기본 모델:

```bash
--model Qwen/Qwen3.5-2B
```

기본 dtype:

```bash
--dtype float16
```

WSL 단일 GPU 실험에서 사용한 GPU memory fraction:

```bash
--gpu-memory-utilization 0.70
```

TAPER 논문식 logical request SLO:

```bash
--ttft-slo 2.0
--tpot-slo 0.05
--parallel-slo-mode logical
```

중요: `--parallel-slo-mode logical`이 현재 논문식 해석에 맞는 기본값이다.
`branch`는 각 branch가 독립적으로 50ms TPOT를 만족해야 하는 훨씬 빡센
진단용 모드다. TAPER는 branch를 defer할 수 있으므로, `branch` 기준으로만
보면 TAPER의 의도된 동작 자체가 SLO 위반처럼 보일 수 있다.

TAPER planner 옵션:

```bash
--taper-latency-per-seq-ms 0.5
--taper-rho 1.0
--taper-utility linear
```

`--enforce-eager`는 torch compile / cudagraph 변수를 줄이고 실험 편차를
단순화하기 위해 사용했다.

## 빠른 Smoke Test

코드가 깨지지 않았는지만 확인하는 짧은 테스트:

```powershell
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/c/Users/young/OneDrive/바탕\ 화면/vllm-src && source ~/.venvs/vllm-taper/bin/activate && VLLM_USE_FLASHINFER_SAMPLER=0 python benchmarks/benchmark_taper_plus.py --mode both --model Qwen/Qwen3.5-2B --dtype float16 --max-model-len 512 --gpu-memory-utilization 0.70 --num-requests 2 --request-rate inf --bon-n 8 --serial-fraction 0.5 --output-len 8 --synthetic-prompt-words 64 --ttft-slo 2.0 --tpot-slo 0.05 --parallel-slo-mode logical --taper-latency-per-seq-ms 0.5 --taper-rho 1.0 --output-dir benchmark_outputs/taper_algorithm1_smoke --enforce-eager --log-level INFO"
```

기대하는 결과:

```text
IRP-EAGER와 TAPER가 둘 다 완료되고,
SLO가 100% 근처이며,
preemption_count가 0이면 smoke는 통과로 본다.
```

## 단일 비교 실험 예시

논문식 throughput trap 패턴이 처음 확인된 설정:

```powershell
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/c/Users/young/OneDrive/바탕\ 화면/vllm-src && source ~/.venvs/vllm-taper/bin/activate && VLLM_USE_FLASHINFER_SAMPLER=0 python benchmarks/benchmark_taper_plus.py --mode both --model Qwen/Qwen3.5-2B --dtype float16 --max-model-len 512 --gpu-memory-utilization 0.70 --num-requests 8 --request-rate 1.0 --bon-n 32 --serial-fraction 0.5 --output-len 128 --synthetic-prompt-words 64 --ttft-slo 2.0 --tpot-slo 0.05 --parallel-slo-mode logical --taper-latency-per-seq-ms 0.5 --taper-rho 1.0 --output-dir benchmark_outputs/taper_algorithm1_logical_lowload_bon32_out128_req8_rate1 --enforce-eager --log-level INFO"
```

기록된 대표 결과:

```text
IRP-EAGER:
  Avg TTFT 0.1085s
  P99 TTFT 0.2659s
  Raw throughput 1607.12 tok/s
  Goodput 414.79 tok/s
  SLO 12.50%

TAPER:
  Avg TTFT 0.0960s
  P99 TTFT 0.2332s
  Raw throughput 1380.90 tok/s
  Goodput 721.58 tok/s
  SLO 37.50%
```

해석:

```text
EAGER가 raw throughput은 더 높지만,
TAPER가 SLO-valid goodput과 SLO attainment를 더 높인다.
이것이 논문에서 말하는 throughput trap 패턴이다.
```

## Grid Sweep: num_requests=8

처음 넓게 훑은 grid:

```text
BoN: 8,16,32,64
request_rate: 0.5,1,2,4
output_len: 128,512,1024
num_requests: 8
repeats: 1
modes: irp_eager, taper
serial_fraction: 0.5
parallel_slo_mode: logical
```

실행 명령:

```powershell
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/c/Users/young/OneDrive/바탕\ 화면/vllm-src && source ~/.venvs/vllm-taper/bin/activate && VLLM_USE_FLASHINFER_SAMPLER=0 python benchmarks/run_taper_plus_sweep.py --compare-mode both --model Qwen/Qwen3.5-2B --dtype float16 --gpu-memory-utilization 0.70 --bon-values 8,16,32,64 --request-rates 0.5,1,2,4 --output-lens 128,512,1024 --num-requests-values 8 --serial-fraction 0.5 --parallel-slo-mode logical --synthetic-prompt-words 64 --taper-latency-per-seq-ms 0.5 --taper-rho 1.0 --timeout-s 2400 --output-root benchmark_outputs/taper_algorithm1_grid_bon_rate_len_req8 --enforce-eager"
```

결과 위치:

```text
benchmark_outputs/taper_algorithm1_grid_bon_rate_len_req8/sweep_summary.csv
benchmark_outputs/taper_algorithm1_grid_bon_rate_len_req8/comparison_compact.csv
benchmark_outputs/taper_algorithm1_grid_bon_rate_len_req8/sweep_report.md
```

요약:

```text
Config pairs: 48
Mode rows: 96
Failed runs: 0

TAPER goodput wins: 14 / 48
TAPER SLO wins:     10 / 48
TAPER raw wins:      9 / 48
```

강한 win은 주로:

```text
BoN=16/32
output_len=512/1024
```

BoN=8은 TAPER regulation이 별로 필요 없는 구간이라 손해가 많고,
BoN=64는 이 local setup에서는 둘 다 SLO가 무너지는 경우가 많다.

## Grid Sweep: num_requests=32, repeats=2

더 빡센 workload:

```text
BoN: 8,16,32,64
request_rate: 0.5,1,2,4
output_len: 128,512,1024
num_requests: 32
repeats: 2
modes: irp_eager, taper
serial_fraction: 0.5
parallel_slo_mode: logical
```

실행 명령:

```powershell
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/c/Users/young/OneDrive/바탕\ 화면/vllm-src && source ~/.venvs/vllm-taper/bin/activate && VLLM_USE_FLASHINFER_SAMPLER=0 python benchmarks/run_taper_plus_sweep.py --compare-mode both --model Qwen/Qwen3.5-2B --dtype float16 --gpu-memory-utilization 0.70 --bon-values 8,16,32,64 --request-rates 0.5,1,2,4 --output-lens 128,512,1024 --num-requests-values 32 --repeats 2 --serial-fraction 0.5 --parallel-slo-mode logical --synthetic-prompt-words 64 --taper-latency-per-seq-ms 0.5 --taper-rho 1.0 --timeout-s 3600 --output-root benchmark_outputs/taper_algorithm1_grid_bon_rate_len_req32_rep2 --enforce-eager"
```

결과 위치:

```text
benchmark_outputs/taper_algorithm1_grid_bon_rate_len_req32_rep2/sweep_summary.csv
benchmark_outputs/taper_algorithm1_grid_bon_rate_len_req32_rep2/comparison_compact_avg.csv
benchmark_outputs/taper_algorithm1_grid_bon_rate_len_req32_rep2/comparison_compact_by_repeat.csv
benchmark_outputs/taper_algorithm1_grid_bon_rate_len_req32_rep2/sweep_report.md
```

요약:

```text
Config pairs: 48
Repeats/config: 2
Mode rows: 192
Failed runs: 0

TAPER goodput wins, repeat-averaged: 16 / 48
TAPER SLO wins, repeat-averaged:     12 / 48
TAPER raw wins, repeat-averaged:      5 / 48

TAPER goodput wins, run-level:       33 / 96
TAPER SLO wins, run-level:           21 / 96
```

주의:

```text
num_requests=32는 단일 WSL GPU + Qwen3.5-2B 환경에서 꽤 과격하다.
많은 config에서 EAGER와 TAPER 둘 다 SLO가 거의 0%까지 무너진다.
따라서 baseline goodput이 0 근처인 huge ratio는 조심해서 해석해야 한다.
```

가장 방어 가능한 구간:

```text
BoN=16
output_len=512/1024
request_rate=1,2,4
```

이 구간은 TAPER가 repeat 평균 기준으로 goodput/SLO를 반복적으로 개선했다.

## 결과 집계 방법

Runner는 `sweep_summary.csv`를 만든다. 이 파일은 mode별 row를 가진다.
즉 같은 config에 대해 `irp_eager` row와 `taper` row가 따로 있다.

비교용 CSV:

```text
comparison_compact.csv
comparison_compact_avg.csv
comparison_compact_by_repeat.csv
```

위 파일들은 별도 집계 스크립트로 생성했다. 없으면 아래처럼 Python으로
다시 만들 수 있다. PowerShell에서 실행할 때는 repo root를 workdir로 두는
것이 경로 인코딩 문제를 줄인다.

```powershell
cd "C:\Users\young\OneDrive\바탕 화면\vllm-src"
python - <<'PY'
import csv, math, statistics
from pathlib import Path

root = Path("benchmark_outputs/taper_algorithm1_grid_bon_rate_len_req32_rep2")
rows = list(csv.DictReader((root / "sweep_summary.csv").open(encoding="utf-8")))

keys = ["repeat", "bon_n", "output_len", "request_rate"]
pairs = {}
for row in rows:
    if row["status"] != "ok":
        continue
    key = tuple(row[k] for k in keys)
    pairs.setdefault(key, {})[row["mode"]] = row

paired = []
for key, modes in pairs.items():
    if "irp_eager" not in modes or "taper" not in modes:
        continue
    e = modes["irp_eager"]
    t = modes["taper"]
    def f(row, name):
        return float(row[name])
    paired.append({
        "repeat": int(float(key[0])),
        "bon": int(float(key[1])),
        "out": int(float(key[2])),
        "rate": float(key[3]),
        "e_goodput": f(e, "user_goodput_tokens_s"),
        "t_goodput": f(t, "user_goodput_tokens_s"),
        "e_slo": f(e, "slo_attainment_rate_pct"),
        "t_slo": f(t, "slo_attainment_rate_pct"),
        "e_raw": f(e, "system_throughput_tokens_s"),
        "t_raw": f(t, "system_throughput_tokens_s"),
    })

print("paired runs:", len(paired))
print("TAPER goodput wins:", sum(r["t_goodput"] > r["e_goodput"] for r in paired))
print("TAPER SLO wins:", sum(r["t_slo"] > r["e_slo"] for r in paired))
PY
```

## 해석 기준

보고할 때는 아래 순서로 보는 것이 안전하다.

1. `slo_attainment_rate_pct`
2. `user_goodput_tokens_s`
3. `avg_ttft_s`, `p99_ttft_s`
4. `system_throughput_tokens_s`
5. `preemption_count`

논문 관점에서 중요한 것은 raw throughput이 아니다. EAGER가 raw throughput을
이기면서도 goodput/SLO가 무너지는 것이 throughput trap이다. 따라서
TAPER가 raw throughput에서 항상 이겨야 하는 것은 아니다.

좋은 TAPER 증거:

```text
EAGER raw throughput >= TAPER raw throughput
TAPER goodput > EAGER goodput
TAPER SLO >= EAGER SLO
```

조심해야 하는 증거:

```text
EAGER goodput이 0에 가까워서 ratio가 무한대처럼 보이는 경우
둘 다 SLO 0%인 경우
BoN=64처럼 local setup 자체가 과부하인 경우
```

## 권장 후속 실험

전체 grid를 계속 키우기보다, 확인된 informative region만 좁혀서 반복하는
것이 낫다.

추천:

```text
BoN: 16
output_len: 512,1024
request_rate: 1,2,4
num_requests: 32
repeats: 3-5
parallel_slo_mode: logical
```

이 구간이 professor-facing figure로 가장 깔끔하다. BoN=32/64는 stress
boundary 용도로 따로 보여주는 것이 좋다.

## 흔한 문제

### FlashInfer sampler JIT 에러

`nvcc`나 FlashInfer JIT 관련 에러가 나면 아래 환경변수를 유지한다.

```bash
VLLM_USE_FLASHINFER_SAMPLER=0
```

### Windows PowerShell에서 Python 실행 실패

Torch/vLLM 환경은 WSL venv에 있다. benchmark는 WSL에서 실행한다.

### 경로 인코딩 문제

Python에서 절대 Windows path를 직접 쓰면 `바탕 화면` 경로가 깨질 수 있다.
집계 스크립트는 repo root로 이동한 뒤 상대 경로를 쓰는 편이 안전하다.

### `gpu_memory_utilization=0.35` 등 너무 낮은 값

Qwen3.5-2B 로딩 후 KV cache memory가 부족해서 실패할 수 있다. 지금 실험은
대부분 `0.70`을 사용했다.

### 긴 실행 시간

`num_requests=32`, `repeats=2`, 전체 grid는 몇 시간 이상 걸린다. 중간에
끊겼다면 sweep runner의 `--resume` 옵션을 사용할 수 있다.

예:

```powershell
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/c/Users/young/OneDrive/바탕\ 화면/vllm-src && source ~/.venvs/vllm-taper/bin/activate && VLLM_USE_FLASHINFER_SAMPLER=0 python benchmarks/run_taper_plus_sweep.py <same args> --resume"
```

