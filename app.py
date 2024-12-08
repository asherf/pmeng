import chainlit as cl
import logging

from helpers import MAIN_MODEL, MODEL_TEMPERATURE, get_user_message, client, get_prompt

_logger = logging.getLogger(__name__)


# async def bail_out(message: cl.Message, message_history, error: str) -> None:
#     message_history.append({"role": "user", "content": error})
#     response_message = cl.Message(content="")
#     await response_message.send()
#     await response_message.update()
#     message_history.append({"role": "assistant", "content": response_message.content})
#     cl.user_session.set("message_history", message_history)


@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [get_prompt()])
    _logger.info(f"Message history: {len(message_history)}")
    msg_to_send = get_user_message(message)
    message_history.append(msg_to_send)
    response_message = cl.Message(content="")
    await response_message.send()
    stream = await client.chat.completions.create(
        messages=message_history, stream=True, model=MAIN_MODEL, temperature=MODEL_TEMPERATURE, max_tokens=4000
    )
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)

    await response_message.update()
    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)
