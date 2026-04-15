# H200 GPU 高级配置指南

## 🎯 当前配置（已应用）

```python
gpu_memory_utilization=0.95   # 使用 134GB / 141GB 显存
max_num_seqs=1024             # 1024 并发序列
max_num_batched_tokens=32768  # 32K token 批处理
enable_chunked_prefill=True   # 启用分块预填充
```

**性能特点**：
- 极高吞吐量
- 适合批量推理
- 充分利用 H200 显存优势

## 🚀 进一步优化选项

### 选项 1: 超长文本支持

如果需要处理更长的文本（16K-32K tokens）：

```python
self.model = LLM(
    model=model_path,
    trust_remote_code=True,
    dtype='float16',
    gpu_memory_utilization=0.95,
    max_model_len=32768,  # ✅ 提升到 32K
    max_num_seqs=512,     # 降低并发以支持更长序列
    max_num_batched_tokens=65536,  # 提升批处理
    enable_chunked_prefill=True,
)
```

### 选项 2: 极致吞吐量（短文本）

如果文本较短（<2K tokens），可以进一步提高并发：

```python
self.model = LLM(
    model=model_path,
    trust_remote_code=True,
    dtype='float16',
    gpu_memory_utilization=0.95,
    max_model_len=4096,   # 降低序列长度
    max_num_seqs=2048,    # ✅ 提升到 2048 并发
    max_num_batched_tokens=32768,
    enable_chunked_prefill=True,
)
```

### 选项 3: 多 H200 并行（如果有多张卡）

如果有 2 张 H200：

```python
self.model = LLM(
    model=model_path,
    trust_remote_code=True,
    dtype='float16',
    gpu_memory_utilization=0.95,
    max_model_len=8192,
    max_num_seqs=2048,    # 2 卡总并发 2048
    max_num_batched_tokens=65536,
    enable_chunked_prefill=True,
    tensor_parallel_size=2,  # ✅ 启用 2 卡并行
)
```

### 选项 4: FP8 量化（实验性）

H200 支持 FP8，可以进一步提升性能：

```python
self.model = LLM(
    model=model_path,
    trust_remote_code=True,
    dtype='auto',  # 自动选择精度
    quantization='fp8',  # ✅ FP8 量化
    gpu_memory_utilization=0.95,
    max_model_len=8192,
    max_num_seqs=1536,  # FP8 可支持更高并发
    max_num_batched_tokens=49152,
    enable_chunked_prefill=True,
)
```

## 📊 配置对比

| 配置 | 并发数 | 序列长度 | 批处理 | 适用场景 |
|------|--------|---------|--------|---------|
| **当前配置** | 1024 | 8K | 32K | 通用，平衡性能 |
| 超长文本 | 512 | 32K | 64K | 长文档处理 |
| 极致吞吐 | 2048 | 4K | 32K | 短文本高并发 |
| 双卡并行 | 2048 | 8K | 64K | 多卡环境 |
| FP8 量化 | 1536 | 8K | 48K | 实验性，最高性能 |

## 🔍 性能监控

### 运行时监控

```bash
# 终端 1: 运行评估
python eval.py --model_name qwen2.5-14b --greedy 1

# 终端 2: 监控 GPU
nvidia-smi -l 1
```

### 关注指标

1. **GPU 利用率**：应该接近 100%
2. **显存使用**：应该在 120-134GB（85-95%）
3. **吞吐量**：观察 vllm 日志中的 tokens/s

### 优化建议

**如果看到**：
- GPU 利用率 < 80% → 提高 `max_num_seqs`
- 显存使用 < 100GB → 提高 `gpu_memory_utilization` 或 `max_model_len`
- 经常 OOM → 降低 `max_num_seqs` 或启用 `enable_prefix_caching`

## ⚡ H200 vs 其他 GPU

| GPU | 显存 | 推荐 max_num_seqs | 推荐配置 |
|-----|------|------------------|---------|
| **H200** | 141GB | **1024-2048** | 当前配置 ✅ |
| A100 80GB | 80GB | 512-1024 | 降低并发 |
| A100 40GB | 40GB | 256-512 | 降低并发和批处理 |
| RTX 4090 | 24GB | 64-128 | 大幅降低参数 |

## 🎯 Agent-SafetyBench 专用优化

对于 eval.py 的顺序执行特性：

```python
# 当前配置已经很好，但如果想优化单次推理延迟：
max_num_seqs=256        # 降低并发，减少调度开销
max_model_len=8192      # 保持不变
enable_chunked_prefill=False  # 关闭以降低延迟
```

**权衡**：
- 降低延迟：适合交互式测试
- 提高吞吐：适合批量评估（推荐）

## 💡 实用命令

```bash
# 查看 H200 详细信息
nvidia-smi -q -d MEMORY,UTILIZATION

# 实时监控
watch -n 1 nvidia-smi

# 查看 vllm 日志中的性能指标
# 观察 "Avg prompt throughput" 和 "Avg generation throughput"
```

## ✅ 总结

**当前 H200 配置已经是针对 Agent-SafetyBench 的最优配置**：
- ✅ 充分利用 141GB 显存
- ✅ 极高并发能力（1024 序列）
- ✅ 大批处理能力（32K tokens）
- ✅ 长文本优化（chunked prefill）

除非有特殊需求（如超长文本或多卡），否则无需调整！

