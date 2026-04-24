# AI Model Selection Assistant

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue)](https://python.org)


Интеллектуальный ассистент для клиентов MWS, который помогает
выбрать подходящую GPT Model Hub‑модель под их продуктовый кейс и
бюджет. 


---

##  Для чего это нужно?

При выборе LLM для production-задач разработчики сталкиваются с десятками моделей от разных провайдеров. Каждая модель имеет десятки характеристик: размер, контекстное окно, цена за токен, скорость инференса, качество на разных доменах.

Ассистент автоматизирует подбор: принимает требования на естественном языке, фильтрует модели по техническим ограничениям, ранжирует по приоритетам пользователя и выдаёт топ-K рекомендаций с объяснением.

---

## 🧠 Архитектура и роли агентов

```
Пользователь → [OpenAI API] → Runner ADK → mws_orchestrator (LLM_Agent)
                                                 │
                                     ┌───────────┼───────────┐
                                     ▼           ▼           ▼
                            get_snapshot  filter_models  score_models
                                     │           │           │
                                     └───────────┼───────────┘
                                                 ▼
                                          ParserCacheAgent
                                                 │
                                                 ▼
                                          MWSTableScraper → mws.ru
```

### Компоненты

**mws_orchestrator** — основной агент на Google ADK (LlmAgent). Принимает запрос пользователя на естественном языке, извлекает структурированные требования, вызывает инструменты фильтрации и ранжирования, формирует финальный ответ.

**SelectionTools** — набор инструментов агента:

- `get_models_snapshot()` - возвращает актуальный список всех моделей из кэша.
- `filter_models_by_requirements` — отсеивает модели по контексту, размеру и бюджету
- `score_models` — ранжирует оставшиеся по формуле, учитывающей приоритеты пользователя (качество, цена, скорость)

**ParserCacheAgent** — кэширующий парсер. Раз в N секунд (по умолчанию 300) обновляет данные о моделях с mws.ru, хранит в памяти в виде pandas DataFrame и списка словарей. Работает асинхронно, не блокирует запросы.

**MWSTableScraper** — низкоуровневый парсер HTML-таблиц. Извлекает характеристики моделей и цены с двух страниц mws.ru, склеивает в единую таблицу.

### Pipeline обработки запроса

1. Пользователь отправляет запрос → агент извлекает из запроса пользователя основные пожелания в словарь `requirements`
2. `filter_models_by_requirements(requirements)` → отсев по контексту, размеру, бюджету
3. `score_models(result)` → ранжирование отфильтрованных моделей
4. Агент возвращает топ-K моделей с объяснением

---

## 📡 OpenAI-совместимый API

Сервер реализован на asyncio без фреймворков, полностью совместим с OpenAI API. 

### Эндпоинты

| Метод | Путь | Описание                        |
|-------|------|---------------------------------|
| GET | `/v1/models` | Список доступных моделей агента |
| POST | `/v1/chat/completions` | Чат-взаимодействие с агентом    |

### Пример запроса и ответа

**Запрос:**

```
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "llama-3.3-70b",
  "messages": [
    {
      "role": "user",
      "content": "Нужна модель для чат-бота поддержки. Бюджет до 5000 руб/мес, 100000 запросов, вход 500 токенов, выход 300. Контекст от 16K. Топ-3."
    }
  ]
}
```

**Ответ (200):**

```json
{
  "id": "chatcmpl-1713950000000",
  "object": "chat.completion",
  "created": 1713950000,
  "model": "llama-3.3-70b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "На основе ваших требований подобраны следующие модели (топ-3): 1. qwen3-32b от QWEN — контекст 40K, размер 32B, стоимость ~3780 руб/мес. Почему: отличный баланс цены и качества, контекст покрывает требование 16K с запасом. 2. deepseek-r1-distill-qwen-32b от DeepSeek — контекст 128K, размер 32B, стоимость ~3780 руб/мес. Почему: огромный контекст для длинной истории диалога, цена как у лидера. 3. gemma-3-27b-it от Google — контекст 128K, размер 27B, стоимость ~3780 руб/мес. Почему: мультимодальный ввод (скриншоты от клиентов), большой контекст, проверенное качество Google. Все модели прошли фильтрацию по контексту (≥16K) и бюджету (≤5000 руб/мес)."
      },
      "finish_reason": "stop"
    }
  ]
}
```

**GET /v1/models:**

```json
{"object": "list",
  "data":
  [
    {"id": "llama-3.3-70b", "object": "model", "created": 1777039497, "owned_by": "huggingface"},
    {"id": "qwen-72b", "object": "model", "created": 1777039497, "owned_by": "huggingface"}, 
    {"id": "groq-llama", "object": "model", "created": 1777039497, "owned_by": "groq"}, 
    {"id": "gemini-flash", "object": "model", "created": 1777039497, "owned_by": "gemini"}
  ]
}
```

---

## 🚀 Локальный запуск

### Вариант 1: Развертывание локального API

```bash
# Клонирование
git clone https://github.com/ArtemOtr/AI_model_selection_assistant.git
cd AI_model_selection_assistant

# Виртуальное окружение
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Зависимости
pip install -r requirements.txt

# Конфигурация
cd agents
cp .env.example .env
# Отредактируйте .env — добавьте нужные вам ключи
# Отредактируйте agents/agents_config.json.
# - Добавьте нужные вам модели
# - Добавьте переменные окружения в поля api_key_env нужных моделей.
# - Выберите модель по умолчанию (default)
cd ..
# Запуск OpenAI-совместимого API
python app.py
```

Сервер запустится на `http://0.0.0.0:8000`.

### Вариант 2: ADK Web (локальный веб-интерфейс для работы с агентом Google ADK)

```bash
git clone https://github.com/ArtemOtr/AI_model_selection_assistant.git

cd AI_model_selection_assistant

adk web
```

Откройте `http://127.0.0.1:8000`, выберите app `agents`.


### Конфигурация моделей (agents/agents_config.json)

Файл содержит пример конфигурации, на основе которой при локальном запуске можно настроить свою конфигурацию под себя.
Модели не от Google передаются в ADK агента с помощью LiteLlm. **Обратите внимание, что не все сторонние модели
смогут использовать правильно инструменты ADK (иметь возможность делать tool-calling)**

```json
{
  "default": "llama-3.3-70b",
  "models": {
    "llama-3.3-70b": {
      "provider": "huggingface",
      "model": "huggingface/together/meta-llama/Llama-3.3-70B-Instruct",
      "api_key_env": "HF_TOKEN"
    }
  }
}
```

- `default` — модель, используемая если в запросе не указана конкретная
- `provider` — `gemini` для Google моделей (возвращается строка), любой другой для LiteLlm
- `api_key_env` — имя переменной из `.env`, где лежит API-ключ

---

## 📁 Структура проекта

```
AI_model_selection_assistant/
├── agents/
│   ├── agent.py              # Точка входа: загрузка конфига, создание агента
│   ├── selection_agent.py    # Агент подбора: инструкция, фильтрация, скоринг
│   ├── parser_agent.py       # Кэширующий парсер таблиц с mws.ru
│   ├── mws_scraper.py        # Низкоуровневый HTML-парсер
│   ├── .env.example          # Пример переменных окружения
│   └── agents_config.json   # Конфигурация доступных LLM-моделей
├── app.py                    # OpenAI-совместимый HTTP-сервер на asyncio
├── requirements.txt          # Python-зависимости
└── README.md                 # Этот файл
```

---

