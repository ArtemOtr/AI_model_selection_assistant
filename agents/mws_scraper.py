import pandas as pd


class MWSTableScraper():
    '''Парсер таблиц с MWS с инфой о моделях и биллингами в pandas'''
    def __init__(self, model_info_url, model_cost_url, cache_ttl_seconds=600):
        self.info_url = model_info_url
        self.cost_url = model_cost_url

    def get_merged_tables(self):
        models_info = pd.read_html(self.info_url)[0]
        models_cost = pd.read_html(self.cost_url)[0]
        merged_models = pd.concat([models_info, models_cost.drop(columns=["Модель"])], axis=1)

        columns_naming = {
            "Параметр": "model",
            "Разработчик": "developer",
            "Формат ввода": "input_format",
            "Формат вывода": "output_format",
            "Контекст, в тысячах токенов": "context_thousands_tokens",
            "Размер модели, в млрд. параметров": "model_size_billion_params",
            "Модель": "model_name",
            "Цена за 1000 входящих токенов, с НДС 22% в период акции с 15 апреля по 15 июля": "promo_input_price_per_1k_tokens",
            "Цена за 1000 исходящих токенов, с НДС 22% в период акции с 15 апреля по 15 июля": "promo_output_price_per_1k_tokens",
            "Цена за 1000 входящих токенов, с НДС 22%": "input_price_per_1k_tokens",
            "Цена за 1000 исходящих токенов, с НДС 22%": "output_price_per_1k_tokens",
            "Отпускная единица, в токенах": "billing_unit_tokens"
        }
        merged_models.rename(columns=columns_naming, inplace=True)

        numeric_cols = [
            'context_thousands_tokens', 'model_size_billion_params',
            'promo_input_price_per_1k_tokens', 'promo_output_price_per_1k_tokens',
            'input_price_per_1k_tokens', 'output_price_per_1k_tokens',
            'billing_unit_tokens'
        ]
        for col in numeric_cols:
            merged_models[col] = pd.to_numeric(
                merged_models[col].replace(['–', '—', '-', ''], pd.NA),
                errors='coerce'
            ).fillna(0.0)

        merged_models['input_format'] = merged_models['input_format'].apply(
            lambda x: [s.strip() for s in str(x).split(',')] if pd.notna(x) else []
        )

        return merged_models

