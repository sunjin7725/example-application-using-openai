import yaml

from openai import OpenAI

from settings import secret_path


class OpenAIClient:
    def __init__(self):
        with open(secret_path, 'r') as f:
            __secret = yaml.safe_load(f)
        __api_key = __secret['openai']['api_key']

        if not hasattr(self, 'client') or self.client is None:
            self.client = OpenAI(api_key=__api_key)

        self.model = 'gpt-4o-mini'

    def __del__(self):
        if hasattr(self, 'client') and self.client is not None:
            self.client.close()
            self.client = None

    def chat(self, messages: list[dict]) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return completion.choices[0].message.content

