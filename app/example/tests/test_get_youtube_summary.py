from example.youtube_summary import get_youtube_summary

if __name__ == "__main__":
    # print(get_youtube_summary("KCuCsVExdLM"))
    test = get_youtube_summary("KCuCsVExdLM")
    # test = get_youtube_summary(video_id="0e2RPUMygEw")
    print(len(test))
    print(test)