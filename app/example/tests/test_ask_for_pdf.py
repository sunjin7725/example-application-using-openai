"""
This file is used to test the ask_for_pdf function.
"""

import os

from settings import data_dir
from client import RedisClient, OpenAIClient
from example.ask_for_pdf import get_intent, generate_response, load_pdf


def load_zelda_rag():
    """
    This function is used to load the zelda pdf into the redis database.
    """
    zelda_pdf_path = os.path.join(data_dir, "zelda_export_guide.pdf")
    load_pdf(zelda_pdf_path)


if __name__ == "__main__":
    redis_client = RedisClient()
    openai_client = OpenAIClient()

    # load_zelda_rag()
    # QUESTION = "Where to find treasure chests?"
    # QUESTION = "어디서 보물상자를 찾아야돼?"
    # QUESTION = "링크가 누구야"
    QUESTION = "1234"
    intent = get_intent(QUESTION)
    facts = redis_client.search_redis(intent)

    print(len(facts))
    print(
        f"""
        INTENT: {intent}
        FACTS: {facts}
    """
    )
    print(generate_response(facts, intent, QUESTION))
