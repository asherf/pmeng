import chainlit as cl
import openai
import base64
import pathlib
import os

api_key = os.getenv("OPENAI_API_KEY")
endpoint_url = "https://api.openai.com/v1"
client = openai.AsyncClient(api_key=api_key, base_url=endpoint_url)

# https://platform.openai.com/docs/models/gpt-4o
model_kwargs = {"model": "gpt-4o-mini", "temperature": 0.3, "max_tokens": 500}



@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])
    images = [fl for fl in message.elements or [] if "image" in fl.mime]
    if images:
        img = base64.b64encode(pathlib.Path(images[0].path).read_bytes()).decode("utf-8")
        message_history.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": message.content or "What’s in this image?",
                    },
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img}"},
                ],
            }
        )
    else:
        message_history.append({"role": "user", "content": message.content})
    response_message = cl.Message(content="")
    await response_message.send()

    stream = await client.chat.completions.create(messages=message_history, stream=True, **model_kwargs)
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)

    await response_message.update()
    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)
