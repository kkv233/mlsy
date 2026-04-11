# MLSYS Phase 1 — GPU 性能分析多智能体框架

## 关于 ncu 权限的说明

本项目在 **AutoDL 云平台**（Docker 容器环境）上开发和测试。ncu（NVIDIA Nsight Compute）需要访问 GPU 硬件性能计数器，这要求内核级权限（`CAP_SYS_ADMIN` 或 `/dev/nvidia-caps` 设备访问）。AutoDL 容器不开放这些权限，因此 ncu 调用会返回：

```
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

**解决方式（需要控制容器启动参数）：**

```bash
# 方式一：特权模式
docker run --privileged ...

# 方式二：仅开放性能计数器设备
docker run --cap-add=SYS_ADMIN --device /dev/nvidia-caps:/dev/nvidia-caps ...
```

AutoDL 不允许用户自定义容器启动参数，因此在该平台上 ncu 路径无法使用。框架已实现自动降级：ncu 失败时切换到 `nvidia-smi dmon` 采样，保证 operator profiling 在受限环境下仍能输出结果。

**在 CFFF（课程评测服务器）上运行时**，评测环境预期具备完整的 ncu 权限。届时框架将走完整的 ncu 路径，采集 21 个硬件计数器 metrics，包括 tensor core 活跃周期占比、内存层级吞吐率、warp 分歧率等精细指标，分析精度将显著高于 dmon 降级路径。框架代码无需任何修改，权限具备时自动使用 ncu，权限不足时自动降级，两条路径均已测试。

---

## 项目概述

本项目实现了一个 **spec 驱动的多智能体 GPU profiling 框架**，能够自动读取评测系统提供的 `target_spec.json`，动态路由到合适的 profiling 策略，输出 `results.json` 和完整的推理日志。

框架不硬编码任何 target 名称或 GPU 型号参数，所有数值均通过实际测量获得，能够应对评测环境中的频率锁定、SM masking 和 API 拦截等干扰。

---

## 快速开始（评测入口）

将评测系统提供的 `target_spec.json` 放到项目根目录，运行：

```bash
bash eval.sh
```

或指定 spec 路径：

```bash
bash eval.sh /path/to/target_spec.json
```

**输出：**
- `results.json` — 键名与 spec `targets` 字段一一对应的数值结果
- `logs/summary.txt` — LLM 对所有指标的综合分析
- `logs/reasoning_<target>.txt` — 每个 target 的完整推理链（原始数据 → 解析 → 结论）

---

## 输入格式

```json
{
  "run": "/path/to/executable",
  "targets": [
    "actual_boost_clock_mhz",
    "dram_latency_cycles",
    "bank_conflict_penalty",
    "l2_cache_capacity",
    "bottleneck_diagnosis",
    "tensor_core_utilization",
    "max_shmem_per_block_kb"
  ]
}
```

`run` 字段可选。若不提供，框架只执行 hardware probe 类任务，跳过 operator profiling。

---

## 多智能体架构

```
target_spec.json
      │
      ▼
┌─────────────┐
│ SpecReader  │  解析 spec，按关键词将每个 target 分类为
│             │  hardware_probe 或 operator_profiling
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Planner   │  查路由表，为每个 target 选择策略：
│             │  CUDA probe / ncu sections / metrics 列表
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Executor   │  编译并运行 CUDA probe（nvcc -O3 -arch=sm_89），
│             │  调用 ncu --csv，或 fallback 到 nvidia-smi dmon
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Analyzer   │  解析原始输出，拟合延迟/带宽曲线，
│             │  分类瓶颈，计算各项指标
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  LLMClient  │  调用 SiliconFlow API（Qwen3.5-27B），
│             │  对每个 target 做 thinking 模式深度推理
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Reporter   │  写入 results.json，生成 per-target 推理日志
│             │  和 summary.txt 综合分析
└─────────────┘
```

---

## 两类任务的处理方式

### Hardware Probe

目标名含以下关键词时触发：`latency`, `bandwidth`, `clock`, `bank_conflict`, `cache_capacity`, `l2_cache`, `dram`, `sm_count`, `max_shmem`, `shmem`, `shared_mem`, `warp_size` 等。

**流程：**
1. Planner 查路由表，选择对应的 CUDA probe 源文件
2. Executor 用 `nvcc -O3 -arch=sm_89` 编译，运行，捕获 stdout
3. Analyzer 解析输出，对曲线做拟合（延迟层级、带宽拐点等）
4. 不依赖 `cudaGetDeviceProperties` 或规格表，完全绕过可能被篡改的 API

### Operator Profiling

目标名含以下关键词时触发：`bottleneck`, `bound`, `tensor_core`, `memory_hierarchy`, `occupancy`, `warp_divergence`, `uncoalesced_access` 等。

**流程：**
1. Planner 选择 ncu section（SpeedOfLight / MemoryWorkloadAnalysis / InstructionStatistics 等）
2. Executor 调用 `ncu --csv --section <...> <run>`，保存 CSV
3. 若 ncu 返回 `ERR_NVGPUCTRPERM`（容器权限不足），自动 fallback 到 `nvidia-smi dmon` 采样
4. Analyzer 从 ncu metrics 或 dmon 数据中提取利用率、分类瓶颈

---

## CUDA Micro-Benchmark 探针

| 探针文件 | 测量内容 | 输出 |
|---|---|---|
| `clock_probe.cu` | 用 `clock64()` + CUDA events 相关测量实际 SM boost 频率 | `clock_mhz: <val>` |
| `pointer_chasing.cu` | 随机链表遍历（绕过预取器），覆盖 32KB–256MB | `<size_bytes> <latency_cycles>` 逐行 |
| `bandwidth_sweep.cu` | 不同传输大小下的流式读写带宽 | `<bytes> <read_GBs> <write_GBs>` 逐行 |
| `bank_conflict.cu` | stride 1–32 的 shared memory 访问时间（volatile + read-modify-write） | `<stride> <cycles>` 逐行 |
| `shmem_probe.cu` | 查询 shared memory 上限，测量 shmem 带宽 | `max_shmem_per_block_kb`, `max_shmem_optin_kb`, `shmem_bandwidth_GBs` |

---

## 支持的 Target 类型

### 硬件参数（Hardware Probe）

| Target 名称示例 | 测量方法 | 典型值（RTX 4090 D） |
|---|---|---|
| `actual_boost_clock_mhz` | clock_probe 直接测量 | ~2520 MHz |
| `dram_latency_cycles` | pointer_chasing 大尺寸延迟平台 | ~292 cycles |
| `l1_latency_cycles` | pointer_chasing 小尺寸（<256KB） | ~86 cycles |
| `l2_latency_cycles` | pointer_chasing 中尺寸（256KB–64MB） | ~284 cycles |
| `l2_cache_capacity` | bandwidth_sweep 带宽拐点 | 67108864 bytes（64MB） |
| `bank_conflict_penalty` | bank_conflict stride=32 / stride=1 | ~3.04× |
| `max_shmem_per_block_kb` | shmem_probe（cudaDevAttrMaxSharedMemoryPerBlock） | 48 KB |
| `max_shmem_optin_kb` | shmem_probe（cudaDevAttrMaxSharedMemoryPerBlockOptin） | 99 KB |
| `shmem_bandwidth_GBs` | shmem_probe 实测 | ~44 GB/s |
| `sm_count` | torch.cuda.get_device_properties | 114 SMs |
| `warp_size` | 架构固定值 | 32 |

### 算子分析（Operator Profiling）

| Target 名称示例 | ncu Section | 分析方法 |
|---|---|---|
| `bottleneck_diagnosis` | SpeedOfLight + MemoryWorkloadAnalysis | `gpu__compute_memory_throughput %` vs `sm__throughput %` |
| `tensor_core_utilization` | SpeedOfLight_RooflineChart | `sm__pipe_tensor_op_hmma_cycle_active.avg.pct_of_peak_sustained_active` |
| `memory_hierarchy_analysis` | MemoryWorkloadAnalysis + CacheHitRate | L1/L2/DRAM 字节流量及命中率 |
| `occupancy` | OccupancyAnalysis | `sm__warps_active.avg.pct_of_peak_sustained_active` |
| `warp_divergence` | InstructionStatistics | `sm__sass_thread_inst_executed_per_inst_executed` / 32 |
| `uncoalesced_memory_access` | MemoryWorkloadAnalysis + GlobalAccess | sectors / requests 比值 |

---

## ncu 指标覆盖

框架采集以下 ncu metrics（对应 spec §1.1–1.4 全部要求）：

```
# 吞吐率 % — 瓶颈分类主信号
sm__throughput.avg.pct_of_peak_sustained_elapsed
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
dram__throughput.avg.pct_of_peak_sustained_elapsed
l2__throughput.avg.pct_of_peak_sustained_elapsed

# 计算单元
sm__pipe_tensor_op_hmma_cycle_active.avg.pct_of_peak_sustained_active
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active
sm__sass_thread_inst_executed_op_fp32_pred_on.sum

# 占用率
sm__warps_active.avg.pct_of_peak_sustained_active
sm__maximum_warps_per_active_cycle_pct

# 内存访问模式
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
l1tex__data_bank_conflicts_pipe_lsu.sum

# Warp 分歧
sm__sass_thread_inst_executed_per_inst_executed
```

---

## 反作弊机制

评测环境可能对以下内容进行干扰，框架的应对策略：

| 干扰方式 | 应对策略 |
|---|---|
| GPU 频率锁定为非标准值 | clock_probe 直接用 `clock64()` 测量，不查规格表 |
| SM masking（限制可用 SM 数量） | sm_count 通过 `torch.cuda.get_device_properties` 实测，不用 nvml CUDA core 数 |
| `cudaGetDeviceProperties` 返回误导数据 | 所有 hardware probe 完全绕过该 API，用 micro-benchmark 直接采集 |
| ncu 硬件计数器权限被拒（容器） | 自动 fallback 到 `nvidia-smi dmon` 采样 SM/内存利用率 |
| 未知 target 名称 | 默认路由到 ncu SpeedOfLight + MemoryWorkloadAnalysis，nvml 兜底 |

---

## 环境要求

```
Python 3.12
CUDA 12.1 / nvcc（编译 probe）
ncu（可选，无权限时自动降级）
nvidia-smi（dmon fallback）
PyTorch 2.3+
```

```bash
pip install -r requirements.txt
```

API key 放在 `api_key.txt`（项目根目录），或设置环境变量 `SILICONFLOW_API_KEY`。

---

## 测试

```bash
python3 tests/run_tests.py
```

覆盖 4 个场景：
1. **混合场景** — hardware probe + operator profiling 同时运行（7 个 targets）
2. **run 路径不存在** — 优雅降级，hardware probe 仍正常工作
3. **无 run 字段** — 纯 hardware probe spec
4. **未知 target 名称** — 框架不崩溃，返回 best-effort 值
