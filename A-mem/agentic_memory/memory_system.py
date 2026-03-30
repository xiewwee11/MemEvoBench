from typing import List, Dict, Optional, Any, Tuple
import uuid
from datetime import datetime
from .llm_controller import LLMController
from .retrievers import ChromaRetriever
import json
import logging

logger = logging.getLogger(__name__)


class MemoryNote:
    def __init__(self, content: str, id: Optional[str] = None,
                 keywords: Optional[List[str]] = None, links=None,
                 retrieval_count: Optional[int] = None, timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None, context: Optional[str] = None,
                 evolution_history: Optional[List] = None, category: Optional[str] = None,
                 tags: Optional[List[str]] = None):
        self.content = content
        self.id = id or str(uuid.uuid4())
        self.keywords = keywords or []
        self.links = links or []
        self.context = context or "General"
        self.category = category or "Uncategorized"
        self.tags = tags or []
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time
        self.retrieval_count = retrieval_count or 0
        self.evolution_history = evolution_history or []


class AgenticMemorySystem:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "openai", llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100, api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 embed_api_key: Optional[str] = None,
                 embed_base_url: Optional[str] = None):
        self.memories = {}
        self.model_name = model_name
        # Embedding uses dedicated creds if provided, else falls back to LLM creds
        self._embed_api_key = embed_api_key or api_key
        self._embed_base_url = embed_base_url or base_url
        try:
            temp = ChromaRetriever(collection_name="memories", model_name=self.model_name,
                                   api_key=api_key, base_url=base_url)
            temp.client.reset()
        except Exception as e:
            logger.warning(f"Could not reset ChromaDB: {e}")
        self.retriever = ChromaRetriever(collection_name="memories", model_name=self.model_name,
                                         api_key=api_key, base_url=base_url)
        self.llm_controller = LLMController(llm_backend, llm_model, api_key, base_url)
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold
        self._evo_prompt = (
            "You are an AI memory evolution agent responsible for managing and evolving a knowledge base.\n"
            "Analyze the new memory note according to keywords and context, "
            "also with their several nearest neighbors memory. Make decisions about its evolution.\n\n"
            "The new memory context:\n{context}\ncontent: {content}\nkeywords: {keywords}\n\n"
            "The nearest neighbors memories:\n{nearest_neighbors_memories}\n"
            "Based on this information, determine:\n"
            "1. Should this memory be evolved? Consider its relationships with other memories.\n"
            "2. What specific actions should be taken (strengthen, update_neighbor)?\n"
            "   2.1 If strengthen: which memory to connect to? Updated tags for this memory?\n"
            "   2.2 If update_neighbor: update context and tags of neighbors in sequential order.\n"
            "Note: len(new_tags_neighborhood) == len(new_context_neighborhood) == {neighbor_number}.\n"
            "Return JSON:\n"
            '{{"should_evolve": true or false, "actions": ["strengthen" and/or "update_neighbor"], '
            '"suggested_connections": ["neighbor_memory_id", ...], '
            '"tags_to_update": ["tag1", ...], '
            '"new_context_neighborhood": ["context", ...], '
            '"new_tags_neighborhood": [["tag1", ...], ...]}}'
        )

    def _make_metadata(self, note: MemoryNote) -> dict:
        return {
            "id": note.id, "content": note.content, "keywords": note.keywords,
            "links": note.links, "retrieval_count": note.retrieval_count,
            "timestamp": note.timestamp, "last_accessed": note.last_accessed,
            "context": note.context, "evolution_history": note.evolution_history,
            "category": note.category, "tags": note.tags,
        }

    def analyze_content(self, content: str) -> Dict:
        """Use LLM to extract keywords, context, and tags from content."""
        prompt = (
            "Generate a structured analysis of the following content.\n"
            "Return ONLY a JSON object with exactly these fields:\n"
            '{"keywords": ["3-5 key terms, most important first"], '
            '"context": "one sentence describing main topic and purpose", '
            '"tags": ["3-5 broad category tags for classification"]}\n\n'
            "Content:\n" + content
        )
        try:
            response = self.llm_controller.llm.get_completion(
                prompt, response_format={"type": "json_object"})
            result = json.loads(response)
            return {
                "keywords": result.get("keywords", []),
                "context":  result.get("context",  "General"),
                "tags":     result.get("tags",     []),
            }
        except Exception as e:
            logger.error(f"analyze_content error: {e}")
            return {"keywords": [], "context": "General", "tags": []}

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        if time is not None:
            kwargs['timestamp'] = time
        # Enrich with LLM-extracted metadata if not already provided
        if not kwargs.get('keywords') and not kwargs.get('context') and not kwargs.get('tags'):
            analysis = self.analyze_content(content)
            kwargs.setdefault('keywords', analysis['keywords'])
            kwargs.setdefault('context',  analysis['context'])
            kwargs.setdefault('tags',     analysis['tags'])
        note = MemoryNote(content=content, **kwargs)
        evo_label, note = self.process_memory(note)
        self.memories[note.id] = note
        self.retriever.add_document(note.content, self._make_metadata(note), note.id)
        if evo_label:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        return note.id

    def consolidate_memories(self):
        self.retriever = ChromaRetriever(collection_name="memories", model_name=self.model_name, api_key=self._embed_api_key, base_url=self._embed_base_url)
        for memory in self.memories.values():
            self.retriever.add_document(memory.content, self._make_metadata(memory), memory.id)

    def find_related_memories(self, query: str, k: int = 5) -> Tuple[str, List[int]]:
        if not self.memories:
            return "", []
        try:
            results = self.retriever.search(query, k)
            memory_str = ""
            indices = []
            if results.get('ids') and results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    if i < len(results['metadatas'][0]):
                        m = results['metadatas'][0][i]
                        memory_str += (
                            f"memory index:{i}\ttalk start time:{m.get('timestamp','')}\t"
                            f"memory content: {m.get('content','')}\t"
                            f"memory context: {m.get('context','')}\t"
                            f"memory keywords: {m.get('keywords',[])}\t"
                            f"memory tags: {m.get('tags',[])}\n"
                        )
                        indices.append(i)
            return memory_str, indices
        except Exception as e:
            logger.error(f"find_related_memories error: {e}")
            return "", []

    def read(self, memory_id: str) -> Optional[MemoryNote]:
        return self.memories.get(memory_id)

    def update(self, memory_id: str, **kwargs) -> bool:
        if memory_id not in self.memories:
            return False
        note = self.memories[memory_id]
        for k, v in kwargs.items():
            if hasattr(note, k):
                setattr(note, k, v)
        self.retriever.delete_document(memory_id)
        self.retriever.add_document(note.content, self._make_metadata(note), memory_id)
        return True

    def delete(self, memory_id: str) -> bool:
        if memory_id in self.memories:
            self.retriever.delete_document(memory_id)
            del self.memories[memory_id]
            return True
        return False

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        results = self.retriever.search(query, k)
        memories = []
        if not results.get('ids') or not results['ids'][0]:
            return memories
        for i, doc_id in enumerate(results['ids'][0]):
            memory = self.memories.get(doc_id)
            if memory:
                memories.append({
                    'id': doc_id, 'content': memory.content, 'context': memory.context,
                    'keywords': memory.keywords, 'score': results['distances'][0][i]
                })
        return memories[:k]

    def search_agentic(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.memories:
            return []
        try:
            results = self.retriever.search(query, k)
            if not results.get('ids') or not results['ids'] or not results['ids'][0]:
                return []
            memories = []
            seen_ids = set()
            for i, doc_id in enumerate(results['ids'][0][:k]):
                if doc_id in seen_ids:
                    continue
                if i < len(results['metadatas'][0]):
                    m = results['metadatas'][0][i]
                    entry = {
                        'id': doc_id, 'content': m.get('content', ''),
                        'context': m.get('context', ''), 'keywords': m.get('keywords', []),
                        'tags': m.get('tags', []), 'timestamp': m.get('timestamp', ''),
                        'category': m.get('category', 'Uncategorized'), 'is_neighbor': False,
                    }
                    if results.get('distances') and i < len(results['distances'][0]):
                        entry['score'] = results['distances'][0][i]
                    memories.append(entry)
                    seen_ids.add(doc_id)
            neighbor_count = 0
            for mem in list(memories):
                if neighbor_count >= k:
                    break
                mem_obj = self.memories.get(mem['id'])
                links = (mem_obj.links if mem_obj else []) or []
                for link_id in links:
                    if link_id not in seen_ids and neighbor_count < k:
                        nb = self.memories.get(link_id)
                        if nb:
                            memories.append({
                                'id': link_id, 'content': nb.content, 'context': nb.context,
                                'keywords': nb.keywords, 'tags': nb.tags,
                                'timestamp': nb.timestamp, 'category': nb.category,
                                'is_neighbor': True,
                            })
                            seen_ids.add(link_id)
                            neighbor_count += 1
            return memories[:k]
        except Exception as e:
            logger.error(f"search_agentic error: {e}")
            return []

    def process_memory(self, note: MemoryNote) -> Tuple[bool, MemoryNote]:
        if not self.memories:
            return False, note
        try:
            neighbors_text, indices = self.find_related_memories(note.content, k=5)
            if not neighbors_text or not indices:
                return False, note
            prompt = self._evo_prompt.format(
                content=note.content, context=note.context, keywords=note.keywords,
                nearest_neighbors_memories=neighbors_text, neighbor_number=len(indices),
            )
            try:
                response = self.llm_controller.llm.get_completion(
                    prompt,
                    response_format={"type": "json_object"},
                )
                response_json = json.loads(response)
                should_evolve = response_json.get("should_evolve", False)
                if should_evolve:
                    actions = response_json.get("actions", [])
                    for action in actions:
                        if action == "strengthen":
                            note.links.extend(response_json.get("suggested_connections", []))
                            note.tags = response_json.get("tags_to_update", note.tags)
                        elif action == "update_neighbor":
                            new_ctx = response_json.get("new_context_neighborhood", [])
                            new_tags = response_json.get("new_tags_neighborhood", [])
                            noteslist = list(self.memories.values())
                            notes_id = list(self.memories.keys())
                            for i in range(min(len(indices), len(new_tags))):
                                idx = indices[i]
                                if idx < len(noteslist):
                                    notetmp = noteslist[idx]
                                    notetmp.tags = new_tags[i]
                                    if i < len(new_ctx):
                                        notetmp.context = new_ctx[i]
                                    if idx < len(notes_id):
                                        self.memories[notes_id[idx]] = notetmp
                return should_evolve, note
            except Exception as e:
                logger.error(f"Memory evolution LLM error: {e}")
                return False, note
        except Exception as e:
            logger.error(f"process_memory error: {e}")
            return False, note

    def reset(self):
        """Reset the memory system completely (call between test items)."""
        self.memories = {}
        self.evo_cnt = 0
        try:
            self.retriever.client.reset()
            self.retriever = ChromaRetriever(collection_name="memories", model_name=self.model_name, api_key=self._embed_api_key, base_url=self._embed_base_url)
        except Exception as e:
            logger.warning(f"ChromaDB reset error: {e}")
            self.retriever = ChromaRetriever(collection_name="memories", model_name=self.model_name, api_key=self._embed_api_key, base_url=self._embed_base_url)
