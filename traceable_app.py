import logging
from typing import cast

import chainlit as cl
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_openai import ChatOpenAI

from helpers import MAIN_MODEL, MODEL_TEMPERATURE, get_user_message
from prompts import CODE_FILE_SUMMARIZER_PROMPTS

_logger = logging.getLogger(__name__)


@cl.on_chat_start
async def on_chat_start():
    model = ChatOpenAI(streaming=True, model=MAIN_MODEL, temperature=MODEL_TEMPERATURE)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                CODE_FILE_SUMMARIZER_PROMPTS["v1"],
            ),
            ("user", "{content}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cast(Runnable, cl.user_session.get("runnable"))  # type: Runnable
    msg_to_send = get_user_message(message)
    msg = cl.Message(content="")
    async for chunk in runnable.astream(
        msg_to_send,
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
