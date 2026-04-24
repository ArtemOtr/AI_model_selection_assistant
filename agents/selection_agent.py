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
    '''
        Класс, предстоавляющий инструменты агенту

        Методы (инструменты агента):
            get_models_snapshot() -> list[dict]
            Возвращает актуальный список всех моделей из кэша.

            filter_models_by_requirements(requirements: dict) -> dict
            Фильтрует модели по требованиям пользователя, суммаризированных агентом

            score_models(payload: dict) -> dict
            Оценивает отфильтрованные модели по трем критериям
    '''

    def __init__(self, parser_agent: ParserCacheAgent):
        self._parser_agent = parser_agent


    def get_models_snapshot(self) -> list[dict[str, Any]]:
        return self._parser_agent.get_models_dict()


    def filter_models_by_requirements(self, requirements: dict[str, Any]) -> dict[str, Any]:

        models = self._parser_agent.get_models_dict()

        filtered = []
        budget = requirements.get("budget_rub_month")
        monthly_requests = requirements.get("monthly_requests")
        avg_in = requirements.get("avg_input_tokens")
        avg_out = requirements.get("avg_output_tokens")

        for m in models:

            context = float(m.get("context_thousands_tokens", 0.0))
            size = float(m.get("model_size_billion_params", 0.0))

            min_context = requirements.get("required_context_k_tokens")
            min_size = requirements.get("min_model_size_b")
            max_size = requirements.get("max_model_size_b")

            # 1. контекст
            if min_context is not None and context < min_context:
                continue

            # 2. размер модели
            if min_size is not None and (size < min_size or size > max_size):
                continue

            if budget is not None and monthly_requests and avg_in and avg_out:
                in_price = float(m.get("input_price_per_1k_tokens", 0.0))
                out_price = float(m.get("output_price_per_1k_tokens", 0.0))


                cost_rub = (monthly_requests * avg_in / 1000) * in_price + \
                           (monthly_requests * avg_out / 1000) * out_price

                if cost_rub > budget:
                    continue

            filtered.append({
                "model": m.get("model"),
                "developer": m.get("developer"),
                "context_k": context,
                "size_b": size,
            })

        return {
            "requirements": requirements,
            "candidates": filtered,
            "total_models": len(models),
            "selected_models": len(filtered),
        }

    def score_models(self, payload: dict[str, Any]) -> dict[str, Any]:

        if "payload" in payload and isinstance(payload["payload"], dict):
            inner = payload["payload"]
            if "requirements" in inner and "candidates" in inner:
                payload = inner

        req = SelectionRequirements(**payload["requirements"])
        candidates = payload["candidates"]

        scored = []

        for c in candidates:
            quality = c["size_b"] / 70.0 + c["context_k"] / 128.0
            latency = 1 / (1 + c["context_k"])

            # Оценка стоимости
            cost = 1.0
            if req.avg_input_tokens and req.avg_output_tokens and req.monthly_requests:
                model_full = None
                for m in self._parser_agent.get_models_dict():
                    if m.get("model") == c["model"]:
                        model_full = m
                        break

                if model_full:
                    in_price = float(model_full.get("input_price_per_1k_tokens", 0.0))
                    out_price = float(model_full.get("output_price_per_1k_tokens", 0.0))

                    total_cost = (
                            (req.monthly_requests * req.avg_input_tokens / 1000) * in_price +
                            (req.monthly_requests * req.avg_output_tokens / 1000) * out_price
                    )
                    cost = 1 / (1 + total_cost)

            final_score = (
                    req.quality_priority * quality +
                    req.cost_priority * cost +
                    req.latency_priority * latency
            )

            c["score"] = round(final_score, 4)
            scored.append(c)

        scored.sort(key=lambda x: x["score"], reverse=True)

        top_k = req.top_k if req.top_k else len(scored)  # если top_k=None, показываем всех

        return {
            "top_k": top_k,
            "results": scored[:top_k],
            "all_scored": scored,
        }



def build_selection_agent(
    parser_agent: ParserCacheAgent,
    model_name: str = "gemini-2.0-flash",
) -> LlmAgent:
    '''Построение агента через Google ADK'''

    tools = SelectionTools(parser_agent)
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm
    instruction = """
Ты агент подбора LLM моделей. По запросу пользователя твоя задача будет предлагать наиболее подходящие ему модели. Алгоритм твоей работы описан ниже

## ТВОЙ ЖЁСТКИЙ PIPELINE:

ШАГ 1 — REQUIREMENTS:
Суммаризиируй и сделай на  основе запроса пользователя словарь requirements в формате Python dict:

{
    task_type, domain,
    budget_rub_month,
    monthly_requests,
    avg_input_tokens,
    avg_output_tokens,
    required_context_k_tokens,
    min_model_size_b,
    max_model_size_b
    quality_priority,
    cost_priority,
    latency_priority,
    top_k
}

ПРАВИЛА:
- только этот словарь
- null если нет данных
---

Если пользователь не говорит конкретных чисел, то попытайся их подобрать. Например, если он говорит, что модель нужна мелнькая, то пиши, что max_model_size_b : 32.
Тоже самое относится к цене. По задаче пользователя попытайся определить сколько ему нужно monthly_requests и avg_input_tokens. Но если ты не уверен, то лучше оставляй null в значениях словаря.
Рассчитай примерно также quality_priority, cost_priority, latency_priority.
Заполняй таким образом если полностью уверен в предлагаемых тобой цифрах.

ШАГ 2:
- вызови filter_models_by_requirements(requirements) и получи список моделей, которые соответствуют пожеланиям пользователя. Список моделей будет находится в поле candidates возвращаемого
словаря

ШАГ 3:
- вызови score_models и получи отранжированный список моделей в поле scored возвращаемого словаря
---
ШАГ 4:
- верни top_k моделей из score_models (если top-k не указано, то выводи все модели, прошедшие фильтр) моделей из score_models
- добавь объяснение почему они лучшие

---

ОГРАНИЧЕНИЯ:
- НЕ придумывай модели
- НЕ фильтруй модели сам
- фильтрацию делает Python (filter_models_by_requirements)
- НЕ делай ranking сам
- ranking делает Python (score_models)
- строго tool-based pipeline

## ПРИМЕР 1: Подбор модели для чат-бота поддержки с ограниченным бюджетом

**Пользователь:** "Нужна модель для чат-бота службы поддержки интернет-магазина. Бюджет до 5000 рублей в месяц. Ожидаем около 100 000 запросов в месяц, средняя длина вопроса 500 токенов, ответа — 300 токенов. Желательно, чтобы контекст был не меньше 16 тысяч токенов. Качество ответов очень важно, скорость тоже имеет значение, но цена вторична — главное уложиться в бюджет. Покажи топ-3 варианта."

**Действия агента:**

1. **Извлечение требований (ШАГ 1):**
   - Пользователь САМ назвал конкретные цифры → используем их как есть, ничего не придумываем
   - «Качество очень важно» → quality_priority = 0.6
   - «Скорость имеет значение» → latency_priority = 0.2
   - «Цена вторична» → cost_priority = 0.2
   - Все цифры даны пользователем → я УВЕРЕН, заполняю всё

```python
{
    "task_type": "chatbot",
    "domain": "customer_support",
    "budget_rub_month": 5000.0,
    "monthly_requests": 100000,
    "avg_input_tokens": 500,
    "avg_output_tokens": 300,
    "required_context_k_tokens": 16.0,
    "min_model_size_b": None,
    "max_model_size_b": None,
    "quality_priority": 0.6,
    "cost_priority": 0.2,
    "latency_priority": 0.2,
    "top_k": 3
}

2. Вызов инструментов (ШАГ 2 и 3):

    filter_models_by_requirements(requirements) → возвращает словарь с отфильтрованными кандидатами.

    score_models(payload) → возвращает ранжированный список моделей в поле results.
    
## ТВОЙ ОТВЕТ:
На основе ваших требований подобраны следующие модели (топ-3):

1. **mistralai/Mistral-Large** от Mistral AI
   - Контекст: 128K токенов
   - Размер: 123B параметров
   - Ожидаемая стоимость в месяц: ~4200 руб.
   - Почему: Лучший баланс качества и цены. Большой запас по контексту позволяет хранить длинную историю диалога, а высокая производительность обеспечивает быстрые ответы. Отлично работает с русским языком.

2. **meta-llama/Llama-3.3-70B-Instruct** от Meta
   - Контекст: 128K токенов
   - Размер: 70B параметров
   - Ожидаемая стоимость: ~3800 руб.
   - Почему: Немного экономичнее первой модели при сравнимом качестве. Широко используется в индустрии, имеет хорошую документацию и поддержку сообщества.

3. **Qwen/Qwen2.5-72B-Instruct** от Alibaba
   - Контекст: 32K токенов
   - Размер: 72B параметров
   - Ожидаемая стоимость: ~3500 руб.
   - Почему: Надёжная модель с открытой лицензией, демонстрирующая отличные результаты в задачах на русском языке. Стоимость самая низкая среди кандидатов, при этом качество остаётся на высоком уровне.

Все перечисленные модели прошли фильтрацию по контексту и бюджету, а их итоговый рейтинг рассчитан с учётом заданных вами приоритетов (качество — 60%, стоимость — 20%, скорость — 20%).

## ПРИМЕР 2: Запрос с минимумом информации

**Пользователь:** "Посоветуй какую-нибудь модель для перевода текстов. Чтобы быстро и недорого."

**Действия агента:**

1. **Извлечение требований (ШАГ 1):**
   - Пользователь сказал «быстро» → повышаем latency_priority до 0.4
   - Пользователь сказал «недорого» → повышаем cost_priority до 0.4
   - Качество implicitly важно для перевода → quality_priority = 0.2
   - Задача «перевод текстов» — обычно это средние объёмы: можно предположить monthly_requests = 50000, avg_input_tokens = 500, avg_output_tokens = 500
   - Конкретных цифр по бюджету нет → budget_rub_month = None
   - Требований к размеру модели нет → min_model_size_b = None, max_model_size_b = None
   - Я УВЕРЕН в цифрах 50000/500/500 для задачи перевода → заполняю их

```python
{
    "task_type": "translation",
    "domain": None,
    "budget_rub_month": None,
    "monthly_requests": 50000,
    "avg_input_tokens": 500,
    "avg_output_tokens": 500,
    "required_context_k_tokens": None,
    "min_model_size_b": None,
    "max_model_size_b": None,
    "quality_priority": 0.2,
    "cost_priority": 0.4,
    "latency_priority": 0.4,
    "top_k": 3
}
2. Вызов инструментов (ШАГ 2 и 3):

    filter_models_by_requirements(requirements) → поскольку большинство фильтров не заданы, метод вернёт всех кандидатов.

    score_models(payload) → проранжирует модели с учётом стандартных весов (качество, цена, скорость).

3. Формирование ответа (ШАГ 4):
По вашему запросу я подобрал несколько подходящих моделей для перевода. Так как вы не указали конкретные цифры по бюджету и объёмам, я использовал стандартные приоритеты (50% качество, 30% цена, 20% скорость) и отобрал универсальных кандидатов:

1. **google/gemma-2-9b-it** от Google
   - Хорошее качество перевода на основные европейские языки.
   - Умеренная стоимость и высокая скорость работы.
   - Подойдёт для большинства задач без жёстких требований.

2. **mistralai/Mistral-7B-Instruct-v0.1** от Mistral AI
   - Сбалансированная модель с поддержкой множества языков.
   - Немного дороже Gemma, но стабильнее на редких языковых парах.

3. **Qwen/Qwen2.5-7B-Instruct** от Alibaba
   - Отличный вариант для переводов с/на китайский и английский.
   - Очень низкая стоимость при достойном качестве.

Рекомендую начать с первой модели, а если потребуется более точный подбор — уточните бюджет, объём запросов или конкретную языковую пару.

"""

    return LlmAgent(
        name="mws_orchestrator",
        model=model_name,
        instruction=instruction,
        tools=[
            tools.filter_models_by_requirements,
            tools.score_models,
        ],
    )

if __name__ == "__main__":
    parser_agent = ParserCacheAgent()
    root_agent = build_selection_agent(parser_agent = parser_agent)