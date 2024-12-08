import logging
import os
import pathlib

import chainlit as cl
import openai
from langsmith.wrappers import wrap_openai

from prompts import CODE_FILE_SUMMARIZER_PROMPTS

MAIN_MODEL = "gpt-4o-mini"
MODEL_TEMPERATURE = 0.2

CODE_FILE_MIME_PREFIXES = ("application/", "text/")

_logger = logging.getLogger(__name__)

api_key = os.getenv("OPENAI_API_KEY")
endpoint_url = "https://api.openai.com/v1"
client = wrap_openai(openai.AsyncClient(api_key=api_key, base_url=endpoint_url))


def get_prompt(version="v1") -> dict:
    return {"role": "system", "content": CODE_FILE_SUMMARIZER_PROMPTS[version]}


def is_code_file_mime(mime: str) -> bool:
    return any(mime.startswith(prefix) for prefix in CODE_FILE_MIME_PREFIXES)


def get_user_message(message: cl.Message) -> dict:
    code_files = [fl for fl in message.elements or [] if is_code_file_mime(fl.mime)]
    if not code_files:
        return {"role": "user", "content": message.content}

    if len(code_files) > 1:
        _logger.warning("I can only process one code file at a time.")
        return {}
    fp = pathlib.Path(code_files[0].path)
    code_file_content = fp.read_text()
    msg = message.content or "Explain the code file: "
    return {
        "role": "user",
        "content": f"{msg}\n\n\n{fp.name}\n{code_file_content}",
    }
