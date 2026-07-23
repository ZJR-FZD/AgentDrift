from openai import OpenAI
import time
import json
import sys

sys.path.append("./model_api")
from .BaseAPI import BaseAPI
from .config import URL, API_KEY

class ClaudeAPI(BaseAPI):
    def __init__(self, model_name, generation_config={}):
        super().__init__(generation_config)
        self.model_name = model_name
        self.client = OpenAI(
            base_url=URL,
            api_key=API_KEY,
        )

    def response(self, messages, tools):
        if not tools:
            tools = None
        for _ in range(10):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    tools=tools,
                    messages=messages,
                    **self.generation_config
                )
                if completion is None or completion.choices is None or getattr(completion.choices[0], 'error', None) is not None:
                    continue
                return completion
            except Exception as e:
                print(e)
                time.sleep(1)

    def generate_response(self, messages, tools):
        completion = self.response(messages, tools)
        print('======================================')
        print(completion)
        print('======================================')
        if completion is None:
            return None

        msg = completion.choices[0].message

        reasoning = getattr(msg, 'reasoning_content', None) or getattr(msg, 'reasoning', None)

        content = msg.content if msg.content else None

        raw_tool_calls = msg.tool_calls or []
        tool_calls = []
        for tc in raw_tool_calls:
            tool_calls.append({
                "tool_call_id": tc.id,
                "tool_name": tc.function.name,
                "arguments": json.loads(tc.function.arguments) if tc.function.arguments else {}
            })

        return {
            "content": content,
            "tool_calls": tool_calls,
            "reasoning": reasoning
        }