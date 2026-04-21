import pandas as pd


class MWSTableScraper():
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


if __name__ == "__main__":
    model_info_url = "https://mws.ru/docs/cloud-platform/gpt/general/gpt-models.html"

    model_cost_url = "https://mws.ru/docs/cloud-platform/gpt/general/pricing.html"
    scraper = MWSTableScraper(model_info_url, model_cost_url)
    df = scraper.get_merged_tables()
    print("ИТОГОВАЯ ТАБЛИЦА (первые 5 строк):")
    print(df.head())
    print("\n" + "=" * 80)
    print("ИНФОРМАЦИЯ О ТИПАХ ДАННЫХ:")
    print(df.dtypes)
    print("\n" + "=" * 80)
    print("ПРОВЕРКА ПРЕОБРАЗОВАНИЙ:")
    print("1. Формат ввода (должен быть списком):")
    for idx, row in df.iterrows():
        print(f"   {row['model']}: {row['input_format']} (тип: {type(row['input_format'])})")
    print("\n2. Пример цен (должны быть float, прочерк → 0.0):")
    print(f"   bge-m3 promo_output_price_per_1k_tokens = {df.loc[df['model']=='bge-m3', 'promo_output_price_per_1k_tokens'].values[0]}")
    print(f"   deepseek-r1-distill-qwen-32b input_price_per_1k_tokens = {df.loc[df['model']=='deepseek-r1-distill-qwen-32b', 'input_price_per_1k_tokens'].values[0]}")
    print("\n3. Контекст и размер модели (float):")
    print(f"   bge-m3 context_thousands_tokens = {df.loc[df['model']=='bge-m3', 'context_thousands_tokens'].values[0]} (тип {type(df.loc[df['model']=='bge-m3', 'context_thousands_tokens'].values[0])})")
    print(f"   llama-3.3-70b-instruct model_size_billion_params = {df.loc[df['model']=='llama-3.3-70b-instruct', 'model_size_billion_params'].values[0]}")


'''
отфильтровать таблицы
написать класс и методы скрапера
преобрзовать все в json
'''