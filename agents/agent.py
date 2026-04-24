import os
import json
from pathlib import Path
from agents.parser_agent import ParserCacheAgent
from agents.selection_agent import build_selection_agent
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv

load_dotenv()


parser_agent1 = ParserCacheAgent()

CONFIG_PATH = Path(__file__).parent.parent / "agents_config.json"
config = json.loads(CONFIG_PATH.read_text())
DEFAULT_MODEL = config.get("default")


def get_model(model_name: str = None):
    """Возвращает модель для ADK. Для Gemini - строку, для остальных - LiteLlm."""
    config = json.loads(CONFIG_PATH.read_text())
    name = model_name or config.get("default")
    cfg = config["models"].get(name)
    if not cfg:
        raise ValueError(f"Модель '{name}' недоступна")
    api_key = os.getenv(cfg["api_key_env"])
    if not api_key:
        raise ValueError(f"Не задан {cfg['api_key_env']} в .env для модели '{name}'")

    if cfg["provider"] == "gemini":
        return cfg["model"]

    return LiteLlm(model=cfg["model"], api_key=api_key)

def get_agent_for_model(model_name: str = None):
    """Создаёт нового агента с указанной LLM и возвращает его."""
    model = get_model(model_name)
    return build_selection_agent(
        parser_agent=parser_agent1,
        model_name=model,
    )

root_agent = get_agent_for_model()