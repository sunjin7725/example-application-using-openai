from youtube_transcript_api import YouTubeTranscriptApi

from client import OpenAIClient

def get_youtube_transcript(video_id: str, languages: list[str] = ["ko"]) -> str:
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
    return transcript

def get_youtube_summary(video_id: str, question: str) -> str:
    transcript = get_youtube_transcript(video_id)
    text_list = [f"{t["start"]}s: {t["text"]}" for t in transcript]
    text = " ".join(text_list)

    client = OpenAIClient()

    prompt_role = """
        You are a helpful assistant.
        You are given a TRANSCRIPT of a youtube video.
        You should answer USER`s question.
    """

    prompt = f"""
        {prompt_role}
        TRANSCRIPT: {text}
        USER: {question}
    """
    return client.chat([{"role": "user", "content": prompt}])