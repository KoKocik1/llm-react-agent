from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from typing import Any, Dict, List


class MyCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        print("### LLM Start ###")
        print(f"Prompt to LLM was: {prompts[0]}")
        print("****************")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        print("### LLM End ###")
        print(f"Response: {response.generations[0][0].text}")
        print("****************")
