from client import OpenAIClient

def assist_jounalist(facts: list[str], tone: str, length_words: int, style: str) -> str:
    client = OpenAIClient()

    facts = ", ".join(facts)
    prompt_role = """
        You are an asssistant for journalists.
        Your task is to write articles, based on the FACTS that are given to you.
        You should respect the instructions: the TONE, LENGTH, and the STYLE.
        You should write in korean and LENGTH should be in korean words.
    """
    prompt = f"""
        {prompt_role}
        FACTS: {facts}
        TONE: {tone}
        LENGTH: {length_words} words
        STYLE: {style}
    """
    return client.chat([
        {
            "role": "user",
            "content": prompt
        }
    ])

