import pytest
from agents.parser_agent import ParserCacheAgent
from agents.selection_agent import SelectionTools, SelectionRequirements


# Моковые данные моделей, чтобы не вызывать ParserAgent и чтобы тесты не зависели от возможных изменений на сайте MWS
MOCK_MODELS = [
    {
        "model": "llama-3.3-70b-instruct",
        "developer": "Meta",
        "context_thousands_tokens": 128,
        "model_size_billion_params": 70,
        "input_price_per_1k_tokens": 0.03,
        "output_price_per_1k_tokens": 0.05,
    },
    {
        "model": "gemma-3-27b-it",
        "developer": "Google",
        "context_thousands_tokens": 128,
        "model_size_billion_params": 27,
        "input_price_per_1k_tokens": 0.01,
        "output_price_per_1k_tokens": 0.02,
    },
    {
        "model": "qwen3-8b",
        "developer": "QWEN",
        "context_thousands_tokens": 8,
        "model_size_billion_params": 8,
        "input_price_per_1k_tokens": 0.005,
        "output_price_per_1k_tokens": 0.01,
    },
    {
        "model": "kimi-k2-instruct",
        "developer": "Moonshot AI",
        "context_thousands_tokens": 128,
        "model_size_billion_params": 1024,
        "input_price_per_1k_tokens": 0.1,
        "output_price_per_1k_tokens": 0.15,
    },
]


@pytest.fixture
def tools(mocker):
    """Мокаем ParserCacheAgent, чтобы не лезть в интернет"""
    mock_agent = mocker.Mock(spec=ParserCacheAgent)
    mock_agent.get_models_dict.return_value = MOCK_MODELS
    return SelectionTools(mock_agent)


class TestFilterModels:
    """Тесты filter_models_by_requirements"""

    def test_filter_by_context(self, tools):
        requirements = {"required_context_k_tokens": 16.0}
        result = tools.filter_models_by_requirements(requirements)
        assert result["selected_models"] == 3
        models = [m["model"] for m in result["candidates"]]
        assert "qwen3-8b" not in models

    def test_filter_by_min_size(self, tools):
        requirements = {"min_model_size_b": 70}
        result = tools.filter_models_by_requirements(requirements)
        assert result["selected_models"] == 2  # llama и kimi
        models = [m["model"] for m in result["candidates"]]
        assert "llama-3.3-70b-instruct" in models
        assert "kimi-k2-instruct" in models

    def test_filter_by_max_size(self, tools):
        requirements = {"max_model_size_b": 30}
        result = tools.filter_models_by_requirements(requirements)
        assert result["selected_models"] == 2  # gemma и qwen
        models = [m["model"] for m in result["candidates"]]
        assert "gemma-3-27b-it" in models
        assert "qwen3-8b" in models

    def test_filter_by_budget(self, tools):
        requirements = {
            "budget_rub_month": 9000,
            "monthly_requests": 100000,
            "avg_input_tokens": 500,
            "avg_output_tokens": 300,
        }
        result = tools.filter_models_by_requirements(requirements)
        models = [m["model"] for m in result["candidates"]]
        assert "kimi-k2-instruct" not in models

    def test_filter_no_requirements(self, tools):
        requirements = {}
        result = tools.filter_models_by_requirements(requirements)
        assert result["selected_models"] == 4  # все проходят

    def test_filter_combined(self, tools):
        requirements = {
            "required_context_k_tokens": 16.0,
            "min_model_size_b": 50,
            "budget_rub_month": 5000,
            "monthly_requests": 100000,
            "avg_input_tokens": 500,
            "avg_output_tokens": 300,
        }
        result = tools.filter_models_by_requirements(requirements)
        assert result["selected_models"] == 1
        assert result["candidates"][0]["model"] == "llama-3.3-70b-instruct"


class TestScoreModels:
    """Тесты score_models"""

    def test_scoring_basic(self, tools):
        payload = {
            "requirements": {
                "quality_priority": 0.6,
                "cost_priority": 0.2,
                "latency_priority": 0.2,
                "top_k": 3,
                "avg_input_tokens": None,
                "avg_output_tokens": None,
                "monthly_requests": None,
            },
            "candidates": [
                {"model": "llama-3.3-70b-instruct", "developer": "Meta", "context_k": 128, "size_b": 70},
                {"model": "gemma-3-27b-it", "developer": "Google", "context_k": 128, "size_b": 27},
                {"model": "qwen3-8b", "developer": "QWEN", "context_k": 8, "size_b": 8},
            ]
        }
        result = tools.score_models(payload)
        assert len(result["results"]) == 3
        # Llama самая большая — должна быть первой
        assert result["results"][0]["model"] == "llama-3.3-70b-instruct"
        # Qwen самая маленькая — последняя
        assert result["results"][-1]["model"] == "qwen3-8b"

    def test_scoring_top_k(self, tools):
        payload = {
            "requirements": {"quality_priority": 0.5, "cost_priority": 0.3, "latency_priority": 0.2, "top_k": 2},
            "candidates": [
                {"model": "llama-3.3-70b-instruct", "developer": "Meta", "context_k": 128, "size_b": 70},
                {"model": "gemma-3-27b-it", "developer": "Google", "context_k": 128, "size_b": 27},
                {"model": "qwen3-8b", "developer": "QWEN", "context_k": 8, "size_b": 8},
            ]
        }
        result = tools.score_models(payload)
        assert len(result["results"]) == 2

    def test_scoring_all_have_scores(self, tools):
        payload = {
            "requirements": {"quality_priority": 0.5, "cost_priority": 0.3, "latency_priority": 0.2, "top_k": 3},
            "candidates": [
                {"model": "llama-3.3-70b-instruct", "developer": "Meta", "context_k": 128, "size_b": 70},
            ]
        }
        result = tools.score_models(payload)
        assert "score" in result["results"][0]
        assert result["results"][0]["score"] > 0

    def test_scoring_empty_candidates(self, tools):
        payload = {
            "requirements": {"top_k": 3},
            "candidates": []
        }
        result = tools.score_models(payload)
        assert result["results"] == []
        assert result["all_scored"] == []