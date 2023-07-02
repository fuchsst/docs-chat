from typing import List, Mapping, Optional
from langchain import PipelineAI
from llama_index import PromptHelper
from langchain.llms.base import LLM
from openllm import DollyV2Config
from pyparsing import Any

model_name = "dolly-v2"
model_id = "databricks/dolly-v2-7b"

prompt_helper = PromptHelper(max_input_size=1024, num_output=512, max_chunk_overlap=64)


class DollyV2(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt_length = len(prompt)
        response = PipelineAI(prompt=prompt, max_new_tokens=prompt_helper.num_output)[0]["generated_text"]

        # only return newly generated tokens
        return response[prompt_length:]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": model_name}

    @property
    def _llm_type(self) -> str:
        return "openllm"

    @property
    def __name__(self) -> str:
        return model_name

    @property
    def config_class(self) -> str:
        return DollyV2Config
