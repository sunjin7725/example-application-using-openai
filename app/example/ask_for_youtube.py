'''
This file is used to ask a question to a youtube video.
'''
from typing import Iterable

from youtube_transcript_api import YouTubeTranscriptApi

from client import OpenAIClient


def get_youtube_transcript(video_id: str, languages: Iterable[str] = ("ko",)) -> list[dict]:
    '''
    This function is used to get the transcript of a youtube video.

    Args:
        video_id: str: The id of the youtube video.
        languages: Iterable[str]: The languages to get the transcript in.
    Returns:
        list[dict]: The transcript of the youtube video.
    '''
    return YouTubeTranscriptApi.get_transcript(video_id, languages=languages)

def get_youtube_summary(video_id: str, question: str) -> str:
    '''
    This function is used to get the summary of a youtube video.

    Args:
        video_id: str: The id of the youtube video.
        question: str: The question to ask the youtube video.
    Returns:
        str: The summary of the youtube video.
    '''
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
