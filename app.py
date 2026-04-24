import asyncio
import json
import logging
import time
from http import HTTPStatus
from typing import List, Dict, Any

from pydantic import BaseModel, Field, ValidationError

from agents.agent import parser_agent1, get_agent_for_model, DEFAULT_MODEL, CONFIG_PATH

from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types as genai_types



# схемы
class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "organization"

class ListModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelObject]

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time()*1000)}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]


session_service = InMemorySessionService()


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mws-api")


async def handle_request(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    try:
        raw_request = await reader.readuntil(b"\r\n\r\n")
        request_line, headers_str = raw_request.split(b"\r\n", 1)
        method, path, _ = request_line.decode().split()

        # сборка заголовков
        headers: Dict[str, str] = {}
        for line in headers_str.decode().split("\r\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()

        body = b""
        content_length = int(headers.get("content-length", 0))
        if content_length > 0:
            body = await reader.readexactly(content_length)

        # Роутинг
        if path == "/v1/models" and method == "GET":
            await handle_models(writer)
        elif path == "/v1/chat/completions" and method == "POST":
            await handle_chat(writer, body)
        else:
            await send_json(writer, HTTPStatus.NOT_FOUND, {"error": "Not found"})
    except Exception as e:
        logger.exception("Unhandled error during request processing")
        await send_json(writer, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(e)})
    finally:
        writer.close()
        await writer.wait_closed()

async def handle_models(writer: asyncio.StreamWriter):
    logger.info("GET /v1/models")
    try:
        models_dict = json.loads(CONFIG_PATH.read_text())["models"]
        data = [
            ModelObject(id=name, owned_by=cfg.get("provider", "unknown"))
            for name, cfg in models_dict.items()
        ]
        resp = ListModelsResponse(data=data)
        await send_json(writer, HTTPStatus.OK, resp.model_dump())
    except Exception as e:
        logger.error(f"Error in /v1/models: {e}")
        await send_json(writer, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Failed to get models"})


async def handle_chat(writer: asyncio.StreamWriter, body: bytes):
    logger.info("POST /v1/chat/completions")
    try:
        json_body = json.loads(body)
        req = ChatCompletionRequest(**json_body)
    except (json.JSONDecodeError, ValidationError) as e:
        await send_json(writer, HTTPStatus.BAD_REQUEST, {"error": f"Invalid request: {e}"})
        return

    # выбор модели
    requested_model = req.model or DEFAULT_MODEL
    logger.info(f"User requested model: {requested_model}")

    # создание агента с нужной LLM
    try:
        agent = get_agent_for_model(requested_model)
    except Exception as e:
        logger.exception("Failed to create agent")
        await send_json(writer, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(e)})
        return

    # создание Runner под этот запрос
    runner = Runner(
        app_name="mws_chat",
        agent=agent,
        session_service=session_service,
    )


    user_message = req.messages[-1].content if req.messages else ""


    try:
        session = await session_service.create_session(
            app_name="mws_chat", user_id="default_user"
        )
        content = genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(text=user_message)],
        )
        final_text = ""
        async for event in runner.run_async(
            user_id="default_user",
            session_id=session.id,
            new_message=content,
        ):
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_text = event.content.parts[0].text
                break

        if not final_text:
            final_text = "Извините, я не смог сформулировать ответ."

        choice = ChatCompletionChoice(
            message=ChatMessage(role="assistant", content=final_text)
        )
        resp = ChatCompletionResponse(model=requested_model, choices=[choice])
        await send_json(writer, HTTPStatus.OK, resp.model_dump())
    except Exception as e:
        logger.exception("Agent error")
        await send_json(writer, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(e)})



async def send_json(writer: asyncio.StreamWriter, status: HTTPStatus, data: Any):
    body = json.dumps(data).encode()
    headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
    }
    await send_headers(writer, status, headers, body)


async def send_headers(writer: asyncio.StreamWriter, status: HTTPStatus, extra_headers: Dict[str, str], body: bytes = None):
    status_line = f"HTTP/1.1 {status.value} {status.phrase}\r\n"
    headers_str = status_line
    for k, v in extra_headers.items():
        headers_str += f"{k}: {v}\r\n"
    headers_str += "\r\n"
    writer.write(headers_str.encode())
    if body:
        writer.write(body)
    await writer.drain()


async def main():
    '''запуск asunc сервера'''
    server = await asyncio.start_server(
        handle_request, host="0.0.0.0", port=8000
    )
    logger.info("Server started on http://0.0.0.0:8000")
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    asyncio.run(main())