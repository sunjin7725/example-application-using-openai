import os 

from client import OpenAIClient, RedisClient

def get_intent(query: str):
    client = OpenAIClient()

    response = client.chat([
        {
            "role": "user", 
            "content": f"""
                Extract the intent of the following QUERY. You have to answer in English.
                If you don't know the answer, just return "None".
                QUERY: {query}
            """
        }
    ])
    return response

def generate_response(facts: list[str], intent: str, question: str):
    client = OpenAIClient()

    response = client.chat([
        {
            "role": "user", 
            "content": f"""
                Based on the FACTS and INTENT, answer the QUESTION. You have to answer QUESTION`s language.
                If INTENT is "None", just return you don't know the answer.
                FACTS: {facts}
                INTENT: {intent}
                QUESTION: {question}
            """
        }
    ])
    return response

def load_pdf(pdf_path: str):
    redis_client = RedisClient()
    openai_client = OpenAIClient()

    data = openai_client.pdf_to_embeddings(pdf_path)
    redis_client.embeddings_to_redis(data)