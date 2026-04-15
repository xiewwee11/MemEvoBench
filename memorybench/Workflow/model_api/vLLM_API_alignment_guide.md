# vLLM 模型对齐官方 API 效果指南

## 概述

要让 vLLM 部署的模型达到官方 API 的效果，需要从以下几个方面进行配置对齐：

## 1. 采样参数对齐

### Qwen 官方 API 默认参数

```python
# Qwen 官方 API 默认参数
official_params = {
    'temperature': 1.0,           # 采样温度
    'top_p': 0.8,                 # 核采样
    'top_k': 0,                   # 不使用 top-k（或设为很大的值）
    'repetition_penalty': 1.0,    # 重复惩罚
    'max_tokens': 2048,           # 最大生成长度
}
```

### vLLM SamplingParams 对齐配置

```python
from vllm import SamplingParams

sampling_params = SamplingParams(
    temperature=1.0,              # 必须对齐
    top_p=0.8,                    # 必须对齐
    top_k=-1,                     # -1 表示不使用 top_k
    repetition_penalty=1.0,       # 必须对齐
    max_tokens=2048,
    
    # 额外的关键参数
    presence_penalty=0.0,         # 出现惩罚，默认 0
    frequency_penalty=0.0,        # 频率惩罚，默认 0
    
    # 停止词
    stop=None,                    # 或者设置特定的停止词
    stop_token_ids=None,
    
    # 其他
    skip_special_tokens=True,     # 跳过特殊 token
    spaces_between_special_tokens=True,
)
```

## 2. vLLM 模型加载参数优化

### 关键配置项

```python
from vllm import LLM

model = LLM(
    model=model_path,
    tokenizer=tokenizer_path,  # 显式指定 tokenizer 路径
    trust_remote_code=True,
    
    # 精度设置（重要！）
    dtype='bfloat16',  # 或 'float16'，建议与官方一致
    # Qwen 官方通常使用 bfloat16，如果官方用 fp16 则改为 'float16'
    
    # 量化设置（如果官方 API 使用量化）
    # quantization=None,  # 或 'awq', 'gptq', 'fp8' 等
    
    # KV cache 设置
    gpu_memory_utilization=0.90,  # 不要设太高，留出空间
    max_model_len=32768,          # 与官方 API 的上下文长度对齐
    
    # 并发设置
    max_num_seqs=256,             # 根据需求调整
    max_num_batched_tokens=None,  # 自动设置
    
    # 优化选项
    enable_chunked_prefill=False,  # 关闭以减少延迟，提高一致性
    enable_prefix_caching=False,   # 关闭可能影响结果的缓存
    
    # 张量并行
    tensor_parallel_size=1,        # 单卡设为 1
    
    # 其他重要参数
    tokenizer_mode='auto',         # 'auto' 或 'slow'
    enforce_eager=False,           # 设为 True 可能更稳定但更慢
    seed=None,                     # 如果需要可复现性，设置固定种子
)
```

## 3. Chat Template 正确应用

### 确保使用官方 Chat Template

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path,
    trust_remote_code=True,
    use_fast=True  # 使用快速 tokenizer（如果可用）
)

# 应用 chat template 时的正确方式
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
]

# 方式 1：无工具调用
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 方式 2：有工具调用
prompt = tokenizer.apply_chat_template(
    messages,
    tools=tools,
    tokenize=False,
    add_generation_prompt=True
)
```

### 注意事项

1. **系统提示处理**：
   - 某些模型的 chat template 可能不支持在有工具时直接传入 system role
   - 可能需要使用 `system=` 参数单独传入

2. **特殊 token**：
   - 确保 tokenizer 的特殊 token 设置正确
   - 检查 `eos_token`, `pad_token`, `bos_token` 等

## 4. 精度和计算优化

### dtype 选择建议

```python
# 优先级（从高到低）：
# 1. 与官方 API 完全一致的 dtype
# 2. bfloat16（推荐，精度好，支持广泛）
# 3. float16（速度快，但可能有精度损失）

# 检查官方 API 使用的精度
# Qwen 系列通常使用 bfloat16
dtype = 'bfloat16'  # 或 'float16'
```

### 量化选项

```python
# 如果官方 API 没有使用量化，vLLM 也不要量化
quantization = None

# 如果需要量化以节省显存：
# - 'awq': 高精度量化（推荐）
# - 'gptq': 常用量化
# - 'fp8': FP8 量化（需要硬件支持）
```

## 5. 推荐的完整配置

### 对于 Qwen3-80B 模型

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# 1. 模型加载配置
model = LLM(
    model="/path/to/Qwen3-Next-80B-A3B-Instruct",
    tokenizer="/path/to/Qwen3-Next-80B-A3B-Instruct",
    trust_remote_code=True,
    dtype='bfloat16',  # 与官方对齐
    
    # 内存和性能
    gpu_memory_utilization=0.90,
    max_model_len=32768,  # Qwen 官方支持 32K
    
    # 关闭可能影响一致性的优化
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
    
    # 张量并行（根据 GPU 数量）
    tensor_parallel_size=2,  # 2 卡
)

# 2. Tokenizer 加载
tokenizer = AutoTokenizer.from_pretrained(
    "/path/to/Qwen3-Next-80B-A3B-Instruct",
    trust_remote_code=True
)

# 3. 采样参数（与官方 API 对齐）
sampling_params = SamplingParams(
    temperature=1.0,
    top_p=0.8,
    top_k=-1,
    repetition_penalty=1.0,
    max_tokens=2048,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    skip_special_tokens=True,
)

# 4. 生成
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

outputs = model.generate([prompt], sampling_params)
response = outputs[0].outputs[0].text
```

## 6. 常见问题和解决方案

### 问题 1：输出质量不一致

**可能原因**：
- 采样参数不一致
- dtype 不同
- 使用了量化

**解决方案**：
```python
# 1. 确保 temperature 和 top_p 完全对齐
# 2. 使用与官方相同的 dtype
# 3. 避免量化（除非官方也用了）
# 4. 检查 repetition_penalty
```

### 问题 2：生成速度慢

**可能原因**：
- `enable_chunked_prefill=True` 在单请求时会慢
- 并发数设置过低

**解决方案**：
```python
# 单次推理优化
enable_chunked_prefill=False
max_num_seqs=1

# 批量推理优化
enable_chunked_prefill=True
max_num_seqs=256
```

### 问题 3：系统提示不生效

**可能原因**：
- Chat template 应用不正确
- 有工具时 system message 被忽略

**解决方案**：
```python
# 方案 1：检查是否需要单独传 system
if tools:
    # 提取 system message
    system_msg = messages[0]['content'] if messages[0]['role'] == 'system' else None
    user_messages = messages[1:] if system_msg else messages
    
    prompt = tokenizer.apply_chat_template(
        user_messages,
        tools=tools,
        system=system_msg,  # 单独传入
        tokenize=False,
        add_generation_prompt=True
    )
else:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
```

## 7. 验证对齐效果

### 测试脚本

```python
import json

# 准备相同的测试输入
test_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "解释一下量子纠缠"}
]

# vLLM 生成
vllm_response = generate_with_vllm(test_messages)

# 官方 API 生成（如果可以访问）
# official_response = generate_with_official_api(test_messages)

# 对比结果
print("vLLM 输出:", vllm_response)
# print("官方 API 输出:", official_response)

# 检查关键指标：
# 1. 输出长度是否相近
# 2. 内容质量是否相似
# 3. 格式是否一致
```

## 8. 性能 vs 一致性权衡

### 追求一致性（推荐用于对比测试）

```python
model = LLM(
    dtype='bfloat16',
    gpu_memory_utilization=0.85,
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
    enforce_eager=False,
    max_num_seqs=1,  # 单次推理
)
```

### 追求性能（推荐用于生产环境）

```python
model = LLM(
    dtype='bfloat16',
    gpu_memory_utilization=0.95,
    enable_chunked_prefill=True,
    enable_prefix_caching=True,
    max_num_seqs=256,  # 高并发
    max_num_batched_tokens=65536,
)
```

## 9. 检查清单

在部署前，请确认：

- [ ] `dtype` 与官方一致（通常是 `bfloat16`）
- [ ] `temperature` 和 `top_p` 与官方对齐
- [ ] `repetition_penalty` 设置正确（默认 1.0）
- [ ] `max_model_len` 设置为官方支持的上下文长度
- [ ] Chat template 应用正确
- [ ] 系统提示在有工具时正确传入
- [ ] 没有使用官方 API 未使用的量化
- [ ] 特殊 token 处理正确

## 10. 参考资源

- [vLLM 官方文档](https://docs.vllm.ai/)
- [Qwen 官方文档](https://github.com/QwenLM/Qwen)
- [SamplingParams 参数说明](https://docs.vllm.ai/en/latest/dev/sampling_params.html)


