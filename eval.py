import openai
import logging
import os
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from prompts import CODE_FILE_SUMMARIZER_PROMPTS

_logger = logging.getLogger(__name__)
api_key = os.getenv("OPENAI_API_KEY")
endpoint_url = "https://api.openai.com/v1"
client = wrap_openai(openai.AsyncClient(api_key=api_key, base_url=endpoint_url))
MAIN_MODEL = "gpt-4o-mini"
MODEL_TEMPERATURE = 0.2


def get_prompt(version="v1") -> dict:
    return {"role": "system", "content": CODE_FILE_SUMMARIZER_PROMPTS[version]}


@traceable
def code_file_summarizer_agent(inputs: dict) -> dict:
    messages = [get_prompt(), *inputs["messages"]]
    result = client.chat.completions.create(model=MAIN_MODEL, messages=messages, temperature=MODEL_TEMPERATURE)
    return {"message": {"role": "assistant", "content": result.choices[0].message.content}}
