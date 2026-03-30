from openai import OpenAI
import time
import json
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from BaseAPI import BaseAPI

class OpenaiAPI(BaseAPI):
    def __init__(self, model_name, generation_config={}):
        super().__init__(generation_config)
        self.model_name = model_name
        self.client = OpenAI(base_url="YOUR_API_URL", api_key="YOUR_API_KEY")
        
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
                print(f'completion: {completion}')
                if completion is None or completion.choices is None:
                    continue
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
            tool_call = tool_calls[0]
            tool_call_id = tool_call.id
            tool_name = tool_call.function.name
            if tool_call.function.arguments:
                arguments = json.loads(tool_call.function.arguments)
            else:
                arguments = {}
            return {'type': 'tool', 'tool_call_id': tool_call_id, 'tool_name': tool_name, 'arguments': arguments}

        ## normal content part
        else:
            content = message.content
            return {'type': 'content', 'content': content}
