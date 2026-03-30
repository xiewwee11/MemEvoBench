# Completion 日志说明

## 功能介绍

每次调用 `QwenAPI.generate_response()` 时，原始的 completion 输出都会被自动保存到日志文件中。

## 日志文件位置

- 目录：`evaluation/completion_logs/`
- 文件名格式：`completions_YYYYMMDD_HHMMSS.jsonl`
- 每次启动模型时会创建一个新的日志文件

## 日志格式

每条日志是一个 JSON 对象，包含以下字段：

```json
{
  "timestamp": "2025-10-08T12:34:56.789",
  "completion": "模型的原始输出文本，包含 <think>、<safe-think>、<tool_call> 等标签",
  "num_messages": 3,
  "has_tools": true,
  "num_tools": 5
}
```

## 查看日志

### 1. 查看最新日志文件的最后 10 条记录
```bash
cd evaluation
python view_completion_logs.py
```

### 2. 查看最新日志文件的最后 N 条记录
```bash
python view_completion_logs.py --max 20
```

### 3. 显示完整的 completion 内容（不截断）
```bash
python view_completion_logs.py --full
```

### 4. 查看指定的日志文件
```bash
python view_completion_logs.py --file ./completion_logs/completions_20251008_123456.jsonl
```

### 5. 列出所有日志文件
```bash
python view_completion_logs.py --list
```

## 日志内容说明

每条日志记录包含：

- **timestamp**: 生成时间（ISO 格式）
- **completion**: 模型的原始输出，包括：
  - `<think>...</think>`: 思考过程
  - `<safe-think>...</safe-think>`: 安全思考过程
  - `<tool_call>...</tool_call>`: 工具调用
  - 以及最终的文本内容
- **num_messages**: 当前对话历史中的消息数量
- **has_tools**: 是否提供了工具
- **num_tools**: 提供的工具数量

## 日志文件管理

日志文件使用 JSONL 格式（每行一个 JSON 对象），便于：
- 流式追加写入
- 逐行读取和处理
- 使用标准工具（如 `jq`）进行查询

### 使用 jq 查询日志（如果安装了 jq）

```bash
# 查看所有包含工具调用的记录
cat completions_*.jsonl | jq 'select(.completion | contains("<tool_call>"))'

# 统计有多少条记录包含 safe-think
cat completions_*.jsonl | jq 'select(.completion | contains("<safe-think>"))' | wc -l

# 提取所有 completion 的长度
cat completions_*.jsonl | jq '.completion | length'
```

## 注意事项

1. 日志文件会随着运行时间不断增大，建议定期清理旧的日志
2. 如果磁盘空间有限，可以在 `QwenAPI.__init__()` 中注释掉日志初始化代码
3. 日志保存失败不会影响正常运行，只会打印警告信息


