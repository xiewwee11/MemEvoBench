from openai import OpenAI
import time
import json
import sys
from .BaseAPI import BaseAPI

class OpenaiAPI(BaseAPI):
    def __init__(self, model_name, base_url=None, api_key=None, generation_config={}):
        super().__init__(generation_config)
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        
        self.sys_prompt = self.without_strict_jsonformat_sys_prompt
        # self.sys_prompt = self.sys_prompt_with_failure_modes_without_strict_jsonformat
        # self.sys_prompt = self.sys_prompt_with_simple_failure_modes_without_strict_jsonformat

    def response(self, messages, tools):
        if not tools: tools = None
        for _ in range(8):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    tools=tools,
                    messages=messages,
                    **self.generation_config
                )
                if completion is None or completion.choices is None:
                    continue
                # print(f"completion: {completion}", file=sys.stderr)
                return completion
            except Exception as e:
                print(e)
                time.sleep(1)

    def generate_response(self, messages, tools):
        completion = self.response(messages, tools)
        # print(f'messages: {messages}\ncompletion: {completion}')

        if completion is None: return None

        # 解析首个 choice 的 message
        message = completion.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)

        ## tool call part：仅当 tool_calls 存在且非空时才当作工具调用
        if tool_calls:
            tools_list = []
            
            for tool_call in tool_calls:
                tool_call_id = tool_call.id
                tool_name = tool_call.function.name
                
                if tool_call.function.arguments:
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                else:
                    arguments = {}
                
                tools_list.append({
                    'tool_call_id': tool_call_id,
                    'tool_name': tool_name,
                    'arguments': arguments
                })
            return  {
        'type': 'tool',
        'tools': tools_list,
        'count': len(tools_list)
    }

        ## normal content part
        else:
            content = message.content
            return {'type': 'content', 'content': content}
