'''
This file is used to ask a question to a pdf file.
'''

from client import OpenAIClient, RedisClient


def get_intent(query: str) -> str:
    '''
    This function is used to get the intent of a query.

    Args:
        query: str: The query to get the intent of.
    Returns:
        str: The intent of the query.
    '''
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

def generate_response(facts: list[str], intent: str, question: str) -> str:
    '''
    This function is used to generate a response to a question.

    Args:
        facts: list[str]: The facts to use to generate the response.
        intent: str: The intent of the question.
        question: str: The question to generate a response to.
    Returns:
        str: The response to the question.
    '''
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

def load_pdf(pdf_path: str) -> None:
    '''
    This function is used to load a pdf file into the redis database.

    Args:
        pdf_path: str: The path to the pdf file to load.
    '''
    redis_client = RedisClient()
    openai_client = OpenAIClient()

    data = openai_client.pdf_to_embeddings(pdf_path)
    redis_client.embeddings_to_redis(data)