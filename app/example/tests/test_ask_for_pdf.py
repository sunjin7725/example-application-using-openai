from client import RedisClient, OpenAIClient
from example.ask_for_pdf import get_intent, generate_response

if __name__ == "__main__":
    redis_client = RedisClient()
    openai_client = OpenAIClient()

    # question = "Where to find treasure chests?"
    # question = "어디서 보물상자를 찾아야돼?"
    # question = "링크가 누구야"
    question = "젤다의 전설에서 젤다가 누구야"
    intent = get_intent(question)
    facts = redis_client.search_redis(intent)

    print(len(facts))
    print(f"""
        INTENT: {intent}
        FACTS: {facts}
    """)
    print(generate_response(facts, intent, question))
