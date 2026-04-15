# MemEvoBench

## Quick Start

### 2.1 QA-style

- `iterative_memory_triplequery_test.py`
	- 非 A-MEM 版本
	- 支持 `original` / `only_safe` / `corrected` / `base_model` 模式
	- 每个样本按 `test_query -> test_query_2 -> test_query_3` 三轮迭代，将模型响应持续写回 memory pool
- `amem_iterative_memory_triplequery_test.py`
	- A-MEM 版本
	- 支持 `original` / `only_safe` / `base_model` 模式
	- 使用 A-MEM 进行记忆检索后再执行三轮 query

### 2.2 Workflow-style（`memorybench/Workflow/*`）

- **Static Memory（不带记忆修正工具）**
	- `memorybench/Workflow/normal/eval_workflow.py`
	- `memorybench/Workflow/amem/eval_workflow_amem.py`
- **Dynamic Memory（带 `correct_memory` 修正工具）**
	- `memorybench/Workflow/normal/eval_workflow_with_modtool.py`
	- `memorybench/Workflow/amem/eval_workflow_with_modtool_amem.py`

其中：

- `normal/*`：常规 memory 检索/拼接流程。
- `amem/*`：引入 A-MEM（agentic memory）检索与组织能力。
- `with_modtool*`：增加记忆修正机制，并输出额外的记忆修正指标。

---

## 3. 输入、输出与指标

### 3.1 QA-style 输入数据

QA 脚本默认输入目录为：

- `memorybench/QA/`

会自动匹配：

- `*_memory_triplequery.json`

例如：`finance_memory_triplequery.json`、`health_memory_triplequery.json`。

### 3.2 Workflow-style 输入数据

上述脚本默认读取当前目录下的：

- `workflow.json`

脚本内部是固定文件名读取（`path = "workflow.json"`）。

### 3.3 输出目录（按脚本不同）

- `iterative_memory_triplequery_test.py`  
	`./eval_basemodel_safe/{model_name}/{mode_suffix}/{category}.json(.jsonl)`
- `amem_iterative_memory_triplequery_test.py`  
	`./eval_amem/{model_name}/{mode_suffix}/{category}.json(.jsonl)`
- `eval_workflow.py`  
	`./evaluation_results--/{model_name}/{...}/gen_res.json`
- `eval_workflow_amem.py`  
	`./evaluation_results_amem/{model_name}/{...}/gen_res.json`
- `eval_workflow_with_modtool.py`  
	`./evaluation_results_dynamic/{model_name}/{...}/gen_res.json`
- `eval_workflow_with_modtool_amem.py`  
	`./evaluation_results_dynamic_amem/{model_name}/{...}/gen_res.json`

同时会输出：

- `gen_log.txt`（过程日志）
- `error_samples.json`（异常样本）
- `exceed_round_samples.json`（超轮次样本）

其中 QA-style 还会额外产出跨类别汇总：

- `overall.json`（按 round 聚合指标）

### 3.4 主要指标

- **ASR**（按三轮 query 汇总的攻击成功率/风险触发率统计）
- **Memory correction 指标**（`with_modtool` 版本）
	- `safe_rate`
	- `F1`
- **QA-style correction 指标**（按脚本与模式）
	- `correction_accuracy`
	- `F1`

---

## 4. 运行前配置（环境变量）

脚本通过环境变量选择模型 API 与评审 API。常见键包括：

- 主模型路由（按模型类型使用其一）
	- `OPENAI_API_KEY` / `OPENAI_BASE_URL`
	- `CLAUDE_API_KEY` / `CLAUDE_BASE_URL`
	- `GEMINI_API_KEY` / `GEMINI_BASE_URL`
	- `DEEPINFRA_API_KEY` / `DEEPINFRA_BASE_URL`
- 评测/反馈模型
	- `JUDGE_API_KEY` / `JUDGE_BASE_URL`
	- `FEEDBACK_API_KEY` / `FEEDBACK_BASE_URL`
- A-MEM 额外（`amem/*` 脚本）
	- `AMEM_EMBED_API_KEY`
	- `AMEM_EMBED_BASE_URL`

---

## 5. 最常用命令（聚焦 eval）

> 建议在仓库根目录执行，或先 `cd` 到对应脚本目录再运行。  
> 下述命令是“用法模板”，你可按模型与资源自行调整参数。

### 5.1 QA-style / 非 A-MEM（三轮迭代）

```bash
python iterative_memory_triplequery_test.py \
	--model gpt-5 \
	--mode base_model \
	--judge-model gpt-5.2 \
	--max-items 10
```

如需启用记忆修正工具与反馈：

```bash
python iterative_memory_triplequery_test.py \
	--model gpt-5 \
	--mode corrected \
	--enable-feedback \
	--max-items 10
```

### 5.2 QA-style / A-MEM（三轮迭代）

```bash
python amem_iterative_memory_triplequery_test.py \
	--model Qwen/Qwen3-Next-80B-A3B-Instruct \
	--amem-llm-model Qwen/Qwen3-Next-80B-A3B-Instruct \
	--amem-embed-model Qwen/Qwen3-Embedding-8B \
	--top-k 3 \
	--mode original \
	--max-items 10
```

### 5.3 Normal / Static

```bash
python memorybench/Workflow/normal/eval_workflow.py \
	--model_name Qwen3-Next-80B-A3B-Instruct \
	--debug --debug_samples 2 \
	--num_workers 4
```

### 5.4 A-MEM / Static

```bash
python memorybench/Workflow/amem/eval_workflow_amem.py \
	--model_name Qwen3-Next-80B-A3B-Instruct \
	--debug --debug_samples 2 \
	--amem_top_k 3 \
	--num_workers 4
```

### 5.5 Normal / Dynamic（带记忆修正）

```bash
python memorybench/Workflow/normal/eval_workflow_with_modtool.py \
	--model_name Qwen3-Next-80B-A3B-Instruct \
	--debug --debug_samples 2 \
	--max_workers 4 \
	--use_feedback
```

### 5.6 A-MEM / Dynamic（带记忆修正）

```bash
python memorybench/Workflow/amem/eval_workflow_with_modtool_amem.py \
	--model_name Qwen3-Next-80B-A3B-Instruct \
	--debug --debug_samples 2 \
	--amem_top_k 5 \
	--max_workers 4 \
	--use_feedback
```


---

## 6. 参数速查

- `--model_name`：被测模型（脚本内有支持列表）。
- `--greedy`：`1` 表示贪心，`0` 表示采样。
- `--debug` / `--debug_samples`：小样本快速试跑。
- `--use_feedback`：启用生成式“人类反馈”并写入记忆池。
- `--num_workers` 或 `--max_workers`：并行进程数（不同脚本参数名不同）。
- `--safety_memory`：使用安全增强的 memory prompt（仅部分脚本）。
- `--amem_top_k`：A-MEM 检索条数。
- `--mode`（QA-style）：`original` / `only_safe` / `corrected` / `base_model`（A-MEM 脚本不含 `corrected`）。
- `--judge-model`（QA-style）：用于判定响应是否被误导。
- `--top-k`（QA-style A-MEM）：A-MEM 检索记忆数。
- `--enable-feedback`（QA-style）：生成恶意用户反馈并回注记忆池。

---

## 7. 结果解读建议（写论文可直接引用）

做横向比较时，建议固定同一数据切分、同一 judge 配置后，比较以下维度：

1. `normal` vs `amem`：检索与组织方式变化对 ASR 的影响。
2. `static` vs `dynamic(with_modtool)`：显式记忆修正对安全性与稳定性的影响。
3. `use_feedback` 开/关：偏置信号注入后，记忆演化风险是否被放大。

可将结论组织为：

- **安全退化幅度**（ASR 变化）；
- **修正有效性**（safe rate / F1）；
- **多轮累计效应**（Round1→Round3 的趋势）。

---
