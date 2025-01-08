'''
This file is used to test the generate_news function.
'''

from example.generate_news import assist_jounalist


if __name__ == "__main__":
    test_set_1 = {
        'facts': ['하늘이 맑다', '녹음이 푸르르다'],
        'tone': 'neutral',
        'length_words': 100,
        'style': 'blogpost'
    }

    test_set_2 = {
        'facts': [
            'A book on ChatGPT has been published last week', 
            'The title is Developing Apps with GPT-4 and ChatGPT',
            'The publisher is O`Reilly'
            ],
        'tone': 'excited',
        'length_words': 50,
        'style': 'news flash'
    }

    # chat_test = assist_jounalist(**test_set_1)
    chat_test = assist_jounalist(**test_set_2)

    print(len(chat_test.split()))
    print(chat_test)
