# MLSYS Phase 1 — GPU 性能分析框架

基于 spec 驱动的多智能体 GPU profiling 框架。读取 `target_spec.json`，自动路由到合适的 profiling 策略，输出 `results.json` 和每个 target 的推理日志。

## 运行方式

```bash
pip install -r requirements.txt

# 可选：启用 LLM 解释层
export SILICONFLOW_API_KEY=your_key

python3 main.py target_spec.json
```

输出：`results.json` 和 `logs/reasoning_<target>.txt`。

## 输入格式

```json
{
  "run": "/path/to/executable",
  "targets": [
    "actual_boost_clock_mhz",
    "dram_latency_cycles",
    "bank_conflict_penalty",
    "l2_cache_capacity",
    "bottleneck_diagnosis"
  ]
}
```

## 多智能体结构

```
target_spec.json
      │
      ▼
┌─────────────┐
│ SpecReader  │  解析 spec，将每个 target 分类为
│             │  hardware_probe 或 operator_profiling
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Planner   │  为每个 target 选择策略：
│             │  probe / ncu / torch_profiler
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Executor   │  编译并运行 CUDA probe（nvcc），
│             │  调用 ncu，或 fallback 到 torch.profiler
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Analyzer   │  解析原始输出，拟合曲线，
│             │  识别性能瓶颈
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Reporter   │  调用 LLM 生成推理说明，
│             │  写入 results.json 和 logs/
└─────────────┘
```

### 两类任务

**Hardware Probe**（目标名含 latency / bandwidth / clock / bank_conflict / cache 等关键词）：
- 编译并运行 `probes/` 下的 CUDA micro-benchmark
- 对输出曲线做拟合，提取硬件参数数值
- 不依赖 `cudaGetDeviceProperties` 或规格表（防止评测环境干扰）

**Operator Profiling**（目标名含 bottleneck / tensor_core / memory_hierarchy 等）：
- 用 `ncu` 对 `run` 指向的可执行文件做 kernel 级 profiling
- 若 ncu 无权限（容器环境常见），自动 fallback 到 `torch.profiler`
- 通过 Roofline 分析判断 compute-bound / memory-bound

### CUDA Probe 说明

| 文件 | 测量内容 |
|---|---|
| `clock_probe.cu` | 用 `clock64()` + CUDA events 测量实际 SM boost 频率 |
| `pointer_chasing.cu` | 随机链表遍历，测量 L1/L2/DRAM 访问延迟层级 |
| `bandwidth_sweep.cu` | 不同传输大小下的内存带宽，从带宽拐点估算 L2 容量 |
| `bank_conflict.cu` | 不同 stride（1~32）下的 shared memory 访问时间，量化 bank conflict 惩罚 |

## 评分指标

评测从两个维度打分：

**数值准确性** — `results.json` 中的值与 ground truth 的接近程度：
- Hardware probe：时钟频率（MHz）、延迟（cycles）、带宽（GB/s）、缓存大小（bytes）、冲突惩罚倍数
- Operator profiling：瓶颈分类（`memory_bound` / `compute_bound`）、利用率比例

**方法论分** — `logs/reasoning_<target>.txt` 会交给 LLM judge 评分：
- 是否为每个 target 选择了合理的 profiling 路径
- 证据链是否完整（原始输出 → 解析值 → 推导结论）
- 是否检测到环境异常（锁频、SM masking、API 返回值被篡改）
- 多种测量策略的结果是否互相印证

## 环境说明

- 容器内 ncu 硬件计数器可能无权限（`ERR_NVGPUCTRPERM`），框架会自动切换到 torch.profiler
- GPU 频率可能被锁定为非标准值，所有测量均通过 micro-benchmark 直接采集，不查规格表
- `cudaGetDeviceProperties` 可能返回误导性数据，probe 完全绕过该 API
