import chainlit as cl
import openai
import base64
import pathlib
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
# https://platform.openai.com/docs/models/gpt-4o
# model_kwargs = {"model": MODEL, "temperature": 0.3, "max_tokens": 500}

# TODO: this will be better w/ regex
CODE_FILE_MIME_PREFIXES = ("application/", "text/")


def is_code_file_mime(mime: str) -> bool:
    return any(mime.startswith(prefix) for prefix in CODE_FILE_MIME_PREFIXES)


def get_prompt(version="v1") -> dict:
    return {"role": "system", "content": CODE_FILE_SUMMARIZER_PROMPTS[version]}


@traceable
def code_file_summarizer_agent(inputs: dict) -> dict:
    messages = [get_prompt(), *inputs["messages"]]
    result = client.chat.completions.create(model=MAIN_MODEL, messages=messages, temperature=MODEL_TEMPERATURE)
    return {"message": {"role": "assistant", "content": result.choices[0].message.content}}


async def bail_out(message: cl.Message, message_history, error: str) -> None:
    message_history.append({"role": "user", "content": error})
    response_message = cl.Message(content="")
    await response_message.send()
    await response_message.update()
    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)


@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [get_prompt()])
    code_files = [fl for fl in message.elements or [] if is_code_file_mime(fl.mime)]
    if code_files:
        if len(code_files) > 1:
            _logger.warning("I can only process one code file at a time.")
            bail_out(message, message_history, "I can only process one code file at a time.")
            return
        code_file_content = pathlib.Path(code_files[0].path).read_text()
        # TODO: bail out if the file is too big.
        message_history.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": message.content or "Explain the code file",
                    },
                    {"type": "text", "text": code_file_content},
                ],
            }
        )
    else:
        message_history.append({"role": "user", "content": message.content})
    response_message = cl.Message(content="")
    await response_message.send()

    stream = await client.chat.completions.create(
        messages=message_history, stream=True, model=MAIN_MODEL, temperature=0.2, max_tokens=300
    )
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)

    await response_message.update()
    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)
