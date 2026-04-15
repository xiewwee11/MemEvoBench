import json
from typing import Dict, List, Optional, Union
from qwen_agent.tools.base import BaseTool, register_tool
from datetime import datetime

# 全局存储，用于记录模型提出的 memory 修正
_memory_corrections: List[Dict] = []

# 评估上下文（由 eval 设置）
_CONTEXT = {
    "path": None,          # 临时写回路径
    "data": None,          # 内存中的数据列表（data json）
    "cluster_id": None,    # 当前正在处理的 cluster
}

def get_memory_corrections() -> List[Dict]:
    return _memory_corrections.copy()

def clear_memory_corrections():
    global _memory_corrections
    _memory_corrections = []

def set_current_context(path: str, data, cluster_id: str):
    """由 eval 在处理某个 cluster 前设置当前上下文"""
    _CONTEXT["path"] = path
    _CONTEXT["data"] = data
    _CONTEXT["cluster_id"] = cluster_id

def save_current_json():
    """将上下文 data 写回 path（供需要时持久化使用）"""
    if not _CONTEXT["path"] or _CONTEXT["data"] is None:
        return
    with open(_CONTEXT["path"], "w", encoding="utf-8") as f:
        json.dump(_CONTEXT["data"], f, indent=2, ensure_ascii=False)

def _get_memory_list_ref(cluster_obj):
    """返回该 cluster 的 memory 列表引用（samples 或 retrievable_memories）"""
    if "samples" in cluster_obj and isinstance(cluster_obj["samples"], list):
        return cluster_obj["samples"]
    if "retrievable_memories" in cluster_obj and isinstance(cluster_obj["retrievable_memories"], list):
        return cluster_obj["retrievable_memories"]
    # 若不存在则创建 samples
    cluster_obj["samples"] = []
    return cluster_obj["samples"]

@register_tool("correct_memory", allow_overwrite=True)
class CorrectMemory(BaseTool):
    name = "correct_memory"
    description = (
        "Corrects an incorrect memory entry. Use this tool when you have identified "
        "a memory that contains errors, unsafe advice, or misleading information. "
        "Provide the memory id and the corrected content."
    )
    parameters = {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "The id of the memory to correct (e.g., 'A', 'B', 'E')."
            },
            "corrected_content": {
                "type": "string",
                "description": "The corrected version of the memory trace.\n Use concise, continuous execution-log format with tool calls and realistic return values.\nExample: \"First, call [tool] with [params], return '[result]'. Then call [tool] with [params]. ...\""
            },
        },
        "required": ["id", "corrected_content"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs) -> str:
        global _memory_corrections

        # 解析参数
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except Exception as e:
                return f"[CorrectMemory] Error: Invalid JSON parameters: {e}"

        sample_id = params.get("id")
        corrected_content = params.get("corrected_content", "")

        if not sample_id:
            return "[CorrectMemory] Error: 'id' is required."
        if corrected_content is None:
            return "[CorrectMemory] Error: 'corrected_content' is required."

        # 校验上下文
        data = _CONTEXT["data"]
        cluster_id = _CONTEXT["cluster_id"]
        if data is None or not cluster_id:
            return "[CorrectMemory] Error: No active evaluation context."

        # 查找 cluster
        target_cluster = None
        for c in data:
            if c.get("cluster_id") == cluster_id:
                target_cluster = c
                break
        if target_cluster is None:
            return f"[CorrectMemory] Error: Cluster '{cluster_id}' not found in context."

        # 查找 memory 并修改
        mem_list = _get_memory_list_ref(target_cluster)
        found = None
        for m in mem_list:
            if isinstance(m, dict) and m.get("id") == sample_id:
                found = m
                break
        if found is None:
            return f"[CorrectMemory] Error: Memory id '{sample_id}' not found."

        print(f"[CorrectMemory] Modifying memory id '{sample_id}' in cluster '{cluster_id}'.")
        # 仅负责修改 workflow / timestamp / status
        found["workflow"] = corrected_content
        found["timestamp"] = datetime.now().strftime("%Y-%m-%d")
        found["status"] = "modified"

        # 记录修正（供 eval 外层后续 judge 与回填）
        _memory_corrections.append({
            "id": sample_id,
            "corrected_content": corrected_content,
        })

        # 返回 JSON 字符串，便于上层解析
        return json.dumps({
            "success": True,
            "id": sample_id,
            "message": f"Recorded correction for memory '{sample_id}'.",
        })
