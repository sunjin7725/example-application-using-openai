'''
This file is used to test the ask_for_youtube function.
'''

from example.ask_for_youtube import get_youtube_summary

if __name__ == "__main__":
    # video_id = "KCuCsVExdLM" # 풍월량 패오엑2 영상
    VIDEO_ID = "XjwZUSqkz3Y"  # 착맨 패오엑2 1장 풀영상

    QUESTION = """
        이 영상에서 제일 재밌는 부분이 어딘지 알려줘. 시분초 포맷으로 그 부분의 시간도 알려줘야돼.
    """
    test = get_youtube_summary(VIDEO_ID, QUESTION)
    print(len(test))
    print(test)
