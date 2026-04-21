from dataclasses import dataclass
from typing import Any

from google.adk.agents import LlmAgent

from agents.parser_agent import ParserCacheAgent


@dataclass
class SelectionRequirements:
    task_type: str | None = None
    domain: str | None = None
    budget_rub_month: float | None = None
    monthly_requests: int | None = None
    avg_input_tokens: int | None = None
    avg_output_tokens: int | None = None
    required_context_k_tokens: float | None = None
    min_model_size_b: float | None = None
    quality_priority: float = 0.5
    cost_priority: float = 0.3
    latency_priority: float = 0.2
    top_k: int = 3


class SelectionTools:
    """
    Инструменты для двухагентного пайплайна:
    1) requirements-agent -> фиксированный словарь требований
    2) deterministic filtering в коде
    3) ranking-agent -> ранжирование и объяснение по shortlist
    """

    def __init__(self, parser_agent: ParserCacheAgent):
        self._parser_agent = parser_agent

    def get_models_snapshot(self) -> list[dict[str, Any]]:
        """Возвращает актуальный словарь моделей из ParserCacheAgent."""
        return self._parser_agent.get_models_dict()

    def _estimate_monthly_cost(self, model: dict[str, Any], req: SelectionRequirements) -> dict[str, Any]:
        in_price = float(model.get("input_price_per_1k_tokens", 0.0) or 0.0)
        out_price = float(model.get("output_price_per_1k_tokens", 0.0) or 0.0)

        monthly_in_tokens = int((req.avg_input_tokens or 0) * (req.monthly_requests or 0))
        monthly_out_tokens = int((req.avg_output_tokens or 0) * (req.monthly_requests or 0))

        input_cost = (monthly_in_tokens / 1000.0) * in_price
        output_cost = (monthly_out_tokens / 1000.0) * out_price
        total_cost = input_cost + output_cost

        return {
            "input_price_per_1k_tokens": in_price,
            "output_price_per_1k_tokens": out_price,
            "monthly_input_tokens": monthly_in_tokens,
            "monthly_output_tokens": monthly_out_tokens,
            "input_cost_rub": round(input_cost, 4),
            "output_cost_rub": round(output_cost, 4),
            "total_cost_rub": round(total_cost, 4),
        }

    def filter_models_by_requirements(self, requirements: dict[str, Any]) -> dict[str, Any]:
        """
        Детерминированный этап пайплайна:
        - отсекает модели по жестким ограничениям
        - без эфемерных скоров
        """
        allowed_keys = SelectionRequirements.__dataclass_fields__.keys()
        req = SelectionRequirements(**{k: v for k, v in requirements.items() if k in allowed_keys})
        models = self._parser_agent.get_models_dict()

        filtered: list[dict[str, Any]] = []
        dropped_by_context = 0
        dropped_by_size = 0
        dropped_by_budget = 0

        for model in models:
            context_k = float(model.get("context_thousands_tokens", 0.0) or 0.0)
            model_size_b = float(model.get("model_size_billion_params", 0.0) or 0.0)

            if req.required_context_k_tokens is not None and context_k < req.required_context_k_tokens:
                dropped_by_context += 1
                continue

            if req.min_model_size_b is not None and model_size_b < req.min_model_size_b:
                dropped_by_size += 1
                continue

            cost = self._estimate_monthly_cost(model=model, req=req)
            if req.budget_rub_month is not None and cost["total_cost_rub"] > req.budget_rub_month:
                dropped_by_budget += 1
                continue

            filtered.append(
                {
                    "model": model.get("model"),
                    "developer": model.get("developer"),
                    "input_format": model.get("input_format"),
                    "output_format": model.get("output_format"),
                    "context_thousands_tokens": context_k,
                    "model_size_billion_params": model_size_b,
                    "billing_unit_tokens": model.get("billing_unit_tokens"),
                    "cost": cost,
                }
            )

        return {
            "requirements": requirements,
            "candidates": filtered,
            "filter_stats": {
                "total_models": len(models),
                "selected_models": len(filtered),
                "dropped_by_context": dropped_by_context,
                "dropped_by_model_size": dropped_by_size,
                "dropped_by_budget": dropped_by_budget,
            },
        }


def build_requirements_adk_agent(
    parser_agent: ParserCacheAgent,
    model_name: str = "gemini-2.0-flash",
) -> LlmAgent:
    """
    Агент 1: извлекает фиксированный словарь требований из запроса пользователя.
    """
    _ = parser_agent  # сохраняем совместимость сигнатуры фабрики
    instruction = (
        "Ты requirements-extractor агент для подбора моделей MWS. "
        "Верни ТОЛЬКО JSON-словарь с ключами: "
        "task_type, domain, budget_rub_month, monthly_requests, avg_input_tokens, "
        "avg_output_tokens, required_context_k_tokens, min_model_size_b, "
        "quality_priority, cost_priority, latency_priority, top_k, missing_fields, assumptions. "
        "Если данных не хватает, ставь null и заполняй missing_fields."
    )

    return LlmAgent(
        name="mws_requirements_agent",
        model=model_name,
        instruction=instruction,
    )


def build_ranking_adk_agent(
    parser_agent: ParserCacheAgent,
    model_name: str = "gemini-2.0-flash",
) -> LlmAgent:
    """
    Агент 2: ранжирует  после фильтрации и объясняет выбор.
    """
    tools = SelectionTools(parser_agent=parser_agent)
    instruction = (
        "Ты ranking-agent для подбора моделей MWS. "
        "На вход получаешь: запрос пользователя и словарь требований от requirements-agent на основе запроса пользователя."
        "Сначала вызови tool filter_models_by_requirements(requirements), где requirements — словарь от первого агента. "
        "Используй candidates из результата tool как единственный shortlist для ранжирования. "
        "Запрещено добавлять модели вне shortlist. "
        "Верни структурированный ответ с блоками: "
        "'Входные данные', 'Рекомендованные модели' (top_k), 'Расчеты', 'Пояснения/ограничения'. "
        "Ранжирование обосновывай приоритетами quality_priority/cost_priority/latency_priority."
    )

    return LlmAgent(
        name="mws_ranking_agent",
        model=model_name,
        instruction=instruction,
        tools=[tools.filter_models_by_requirements],
    )