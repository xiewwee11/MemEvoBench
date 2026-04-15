# QwenAPI vllm 并发配置指南

## 🎯 配置参数说明

### 关键并发参数

| 参数 | 默认值 | 作用 | 影响 |
|------|--------|------|------|
| `max_num_seqs` | 256 | 同时处理的序列数 | 并发能力 ↑ |
| `max_num_batched_tokens` | auto | 批处理token数 | 吞吐量 ↑ |
| `gpu_memory_utilization` | 0.9 | GPU显存使用率 | 并发能力 ↑ |
| `max_model_len` | 8192 | 最大序列长度 | 支持的文本长度 |
| `enable_chunked_prefill` | False | 分块预填充 | 长文本并发 ↑ |

## 📊 不同场景的推荐配置

### 场景 1: 极致并发（大显存 GPU，如 A100 80GB）
```python
self.model = LLM(
    model=model_path,
    trust_remote_code=True,
    dtype='float16',
    gpu_memory_utilization=0.95,
    max_model_len=8192,
    max_num_seqs=1024,  # 超高并发
    max_num_batched_tokens=32768,
    enable_chunked_prefill=True,
)
```
- **适用**：A100 80GB, H100
- **并发能力**：极高
- **延迟**：中等

### 场景 2: 平衡模式（中等显存 GPU，如 A100 40GB）
```python
self.model = LLM(
    model=model_path,
    trust_remote_code=True,
    dtype='float16',
    gpu_memory_utilization=0.9,
    max_model_len=8192,
    max_num_seqs=256,  # 当前默认配置
    max_num_batched_tokens=8192,
)
```
- **适用**：A100 40GB, A6000
- **并发能力**：高
- **延迟**：低

### 场景 3: 低延迟优先（小显存或需要快速响应）
```python
self.model = LLM(
    model=model_path,
    trust_remote_code=True,
    dtype='float16',
    gpu_memory_utilization=0.8,
    max_model_len=4096,
    max_num_seqs=64,  # 降低并发
    max_num_batched_tokens=4096,
)
```
- **适用**：RTX 3090, RTX 4090
- **并发能力**：中等
- **延迟**：极低

### 场景 4: 显存受限（小显存 GPU）
```python
self.model = LLM(
    model=model_path,
    trust_remote_code=True,
    dtype='float16',
    gpu_memory_utilization=0.7,
    max_model_len=2048,
    max_num_seqs=32,
    max_num_batched_tokens=2048,
)
```
- **适用**：RTX 3080, V100 16GB
- **并发能力**：低
- **延迟**：低

## 🔧 如何选择配置

### 1. 根据显存大小
```
GPU 显存 -> max_num_seqs 建议值
16GB     -> 32-64
24GB     -> 64-128
40GB     -> 128-256
80GB     -> 256-1024
```

### 2. 根据任务类型
- **短文本高并发**：提高 `max_num_seqs`，降低 `max_model_len`
- **长文本处理**：启用 `enable_chunked_prefill`
- **实时交互**：降低 `max_num_seqs`，获得更低延迟

### 3. 根据 eval.py 的使用场景
```python
# eval.py 是单线程顺序执行，所以：
# - max_num_seqs 不需要太高（256 足够）
# - 更关注单次推理速度
# - 建议配置：max_num_seqs=128-256
```

## 📈 性能监控

运行时观察：
```bash
# 监控 GPU 使用率
nvidia-smi -l 1

# 观察 vllm 日志
# 查看 "GPU blocks" 和 "CPU blocks" 数量
```

如果看到：
- `GPU blocks` 很少 → 提高 `gpu_memory_utilization`
- 经常出现 OOM → 降低 `max_num_seqs` 或 `max_model_len`
- GPU 利用率低 → 提高 `max_num_seqs`

## 🎯 当前配置 (QwenAPI.py)

```python
max_num_seqs=256              # 适中的并发数
max_num_batched_tokens=8192   # 与 max_model_len 相同
gpu_memory_utilization=0.9     # 90% 显存使用
```

**适用场景**：大多数评估任务，平衡性能和稳定性

## 💡 调优建议

1. **从保守配置开始**，逐步增加参数
2. **监控显存使用**，确保不会 OOM
3. **测试实际场景**，观察延迟和吞吐量
4. **根据 GPU 型号**调整配置

## ⚠️ 注意事项

- `max_num_seqs` 过高可能导致 OOM
- `gpu_memory_utilization > 0.95` 可能不稳定
- 启用 `enable_chunked_prefill` 会增加一些开销
- eval.py 是单线程，不需要极高并发配置

