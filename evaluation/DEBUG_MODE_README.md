# Debug 模式使用说明

## 概述
Debug 模式允许你快速测试少量样本，用于验证 API 配置和代码逻辑是否正确。

## 使用方法

### 1. 基本 Debug 模式（测试2个样本）

```bash
cd d:\Agent-SafetyBench\evaluation

# 测试 2 个样本（默认）
python eval.py --debug 1 --model_name qwen3-80B
```

### 2. 自定义测试样本数量

```bash
# 测试 5 个样本
python eval.py --debug 1 --debug_samples 5 --model_name qwen3-80B

# 测试 10 个样本
python eval.py --debug 1 --debug_samples 10 --model_name qwen3-80B
```

### 3. 完整命令示例

```bash
# Debug 模式 + qwen-max 模型 + 贪心解码
python eval.py --debug 1 --debug_samples 2 --model_name qwen3-80B --greedy 1

# Debug 模式 + qwen-plus 模型 + 随机采样
python eval.py --debug 1 --debug_samples 3 --model_name qwen2.5-7b --greedy 0
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--debug` | int | 0 | 是否启用 Debug 模式（0=否，1=是）|
| `--debug_samples` | int | 2 | Debug 模式下测试的样本数量 |
| `--model_name` | str | qwen3-80B | 模型名称 |
| `--greedy` | int | 1 | 是否使用贪心解码 |
| `--extra_info` | str | "" | 额外信息标识 |
| `--allow_empty` | int | 0 | 是否允许空响应 |

## Debug 模式特点

### ✅ 优点
1. **快速验证**：只测试少量样本，快速验证配置
2. **独立输出**：结果保存在 `evaluation_results/tot-{model_name}_debug/` 目录
3. **详细信息**：显示每个测试样本的 ID、指令和风险类别
4. **不影响正式结果**：Debug 输出与正式输出分离

### 📊 输出信息
Debug 模式会显示：
- 🐛 Debug 模式标识
- 原始数据集大小
- 已完成样本数
- 待测试样本列表（ID、指令、风险）
- 处理进度条

### 📁 输出目录结构
```
evaluation_results/
└── tot-qwen3-80B_debug/          # Debug 模式输出
    ├── gen_res.json              # 生成结果
    ├── error_samples.json        # 错误样本
    ├── exceed_round_samples.json # 超轮数样本
    └── gen_log.txt               # 日志文件
```

## 使用流程建议

### 第一步：设置 API Key
编辑 `eval.py` 第 14 行：
```python
QWEN_API_KEY = "sk-your-api-key-here"  # 替换为你的真实 API key
```

### 第二步：Debug 模式测试
```bash
# 先测试 2 个样本
python eval.py --debug 1 --debug_samples 2 --model_name qwen3-80B
```

### 第三步：检查结果
查看输出目录中的文件：
```bash
cat evaluation_results/tot-qwen3-80B_debug/gen_res.json
```

### 第四步：正式运行
确认 Debug 测试成功后，运行完整评估：
```bash
# 关闭 Debug 模式（--debug 0 或不指定）
python eval.py --model_name qwen3-80B
```

## 示例输出

### Debug 模式启用时
```
======================================================================
🐛 DEBUG 模式已启用
   将只测试 2 个样本
======================================================================

初始化 Qwen API 客户端...
完成初始化
Model: qwen3-80B
Generation config: {...}

配置信息:
  数据路径: /path/to/data.json
  输出目录: ./evaluation_results/tot-qwen3-80B_debug
  模型: qwen3-80B
  Debug模式: 是
  测试样本数: 2

原始数据集大小: 1000
已完成样本数: 0

======================================================================
🐛 DEBUG 模式：只测试前 2 个样本
======================================================================
待测试样本:
  1. ID: sample_001
     指令: 请帮我...
     风险: Illegal activity
  2. ID: sample_002
     指令: 如何...
     风险: Privacy violation
======================================================================

Processing samples: 100%|████████████████████| 2/2 [00:30<00:00, 15.2s/it]

success count: 2 (exceed count: 0), fail count: 0
```

## 故障排查

### 问题1：API Key 错误
**症状**：`ValueError: 请通过 api_key 参数传入密钥`  
**解决**：检查 eval.py 第 14 行的 QWEN_API_KEY 是否正确设置

### 问题2：数据文件不存在
**症状**：`❌ 错误：数据文件不存在`  
**解决**：检查 eval.py 第 480 行的 path 变量是否指向正确的文件

### 问题3：网络连接失败
**症状**：`API 调用错误`  
**解决**：检查网络连接，确保可以访问 `dashscope.aliyuncs.com`

## 常见问题

**Q: Debug 模式和正式模式的区别？**  
A: Debug 模式只测试少量样本，输出到独立目录；正式模式测试全部样本。

**Q: 可以指定测试哪些样本吗？**  
A: 当前版本测试前 N 个样本。如需指定特定样本，可以修改代码或使用 test_real_sample.py。

**Q: Debug 结果会覆盖正式结果吗？**  
A: 不会。Debug 模式使用独立的输出目录 (`_debug` 后缀)。

## 相关脚本

- `test_single_sample.py`: 测试简单对话和工具调用
- `test_real_sample.py`: 从数据集中测试单个样本
- `eval.py`: 完整评估脚本（支持 Debug 模式）


