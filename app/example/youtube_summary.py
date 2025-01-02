from youtube_transcript_api import YouTubeTranscriptApi

from client import OpenAIClient

def get_youtube_transcript(video_id: str, languages: list[str] = ["ko"]) -> str:
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
    return transcript

def get_youtube_summary(video_id: str) -> str:
    transcript = get_youtube_transcript(video_id)
    # text_list = [t[""]: t["text"] for t in transcript]
    text_list = [f"{t["start"]}s: {t["text"]}" for t in transcript]
    text = " ".join(text_list)

    client = OpenAIClient()

    prompt_role = """
        You are a helpful assistant.
        You are given a TRANSCRIPT of a youtube video.
        You should answer USER`s question.
    """
    # question = """
    #     First, You should summarize by korean the transcript in 100 words.
    #     Second, Find the best funny part and why that part is funny and print what time(format is hour:minute:second) it is in the transcript.
    # """
    # question = """
    #     영상 안에서 침착맨이 몇번이나 지오너 백작을 언급하는 지 알려줘.
    # """
    question = """
        이 영상에서 제일 재밌는 부분이 어딘지 알려줘. 시분초 포맷으로 그 부분의 시간도 알려줘야돼.
    """

    prompt = f"""
        {prompt_role}
        TRANSCRIPT: {text}
        USER: {question}
    """
    return client.chat([{"role": "user", "content": prompt}])

if __name__ == "__main__":
    # video_id = "KCuCsVExdLM" # 풍월량 패오엑2 영상
    video_id = "XjwZUSqkz3Y" # 착맨 패오엑2 1장 풀영상
    
    test = get_youtube_summary(video_id)
    print(len(test))
    print(test)