"""
This is an example of how to use the OpenAI API to ask a question using a microphone.
"""

from typing import Iterable

import whisper
import gradio as gr

from client import OpenAIClient


STARTING_PROMPT = """
    You are a helpful assistant.
    You can discuss with the user, or perform some actions like sending an email.
    If user ask you to send an email, you have to ask for the subject, recipient, and message.
    You will receive either intructions starting with [INSTRUCTION],  or user input starting with [USER].
    Follow the [INSTRUCTION] and respond to the [USER].
"""

prompts = {
    "START": """
        [INSTRUCTION]
            Write "WRITE_EMAIL" if the user wants to write an email, 
            "QUESTION" if the user has a precise question, 
            "OTHER"  in any other case. Only write one word.
            "EXIT" if the user wants to exit the conversation.
    """,
    "QUESTION": """
        [INSTRUCTION]
            If you can answer the question: ANSWER,
            if you need more information: MORE,
            if you cannot answer: OTHER. Only answer one word.
    """,
    "ANSWER": """
        [INSTRUCTION]
            Answer the [USER]`s question
    """,
    "MORE": """
        [INSTRUCTION]
            Ask the user for more information as specified by previous intructions
    """,
    "OTHER": """
        [INSTRUCTION]
            Give a polite answer or greetings if the user is making polite conversation. 
            Else, answer to the user that you cannot answer the question or do the action
    """,
    "WRITE_EMAIL": """
        [INSTRUCTION]
           If the subject or recipient or body is missing, answer "MORE". 
           Else if you have all the information answer 
           "ACTION_WRITE_EMAIL | subject:subject, recipient:recipient, message:message".
    """,
    "ACTION_WRITE_EMAIL": """
        [INSTRUCTION]
            The mail has been sent. 
            Answer to the user to  tell the action is done
    """,
}

actions = ["ACTION_WRITE_EMAIL"]


class Chat:
    """
    This class is used to chat with the user.
    """

    def __init__(
        self,
        state: str = "START",
        history: Iterable[dict] = ({"role": "user", "content": STARTING_PROMPT},),
    ) -> None:
        self.previous_state = None
        self.state = state
        self.history = history
        self.client = OpenAIClient()
        self.stt_model = whisper.load_model("base")

    def reset(self):
        """
        This function is used to reset the chat.
        """
        self.previous_state = None
        self.state = "START"
        self.history = [{"role": "user", "content": STARTING_PROMPT}]

    def reset_to_previous_state(self):
        """
        This function is used to reset the chat to the previous state.
        """
        self.state = self.previous_state
        self.previous_state = None

    def to_state(self, state: str):
        """
        This function is used to change the state.

        Args:
            state: The state to change to.
        """
        self.previous_state = self.state
        self.state = state

    def do_action(self, action: str) -> str:
        """
        This function is used to do the action.

        Args:
            action: The action to do.

        Returns:
            The result of the action.
        """
        print(f"DEBUG perform action={action}")

    def transcribe_audio(self, audio_path: str) -> str:
        """
        This function is used to transcribe the audio file.

        Args:
            audio_path: The path to the audio file.

        Returns:
            The transcribed text.
        """
        print(f"Transcribing audio file: {audio_path}")
        result = self.stt_model.transcribe(audio_path)
        return result["text"]

    def discuss(self, user_input: str = None) -> str:
        """
        This function is used to continue the conversation.

        Args:
            user_input: The user input. If None, just use the action prompts.

        Returns:
            The response of the conversation.
        """
        if user_input:
            self.history.append({"role": "user", "content": "[USER]\n  " + user_input})
        print(self.history)
        complete_messages = self.history + [{"role": "user", "content": prompts[self.state]}]
        response = self.client.chat(complete_messages)

        # If the response is in prompts, change the state
        if response in prompts:
            self.to_state(response)
            return self.discuss()
        elif response.split("|")[0].strip() in actions:
            action = response.split("|")[0].strip()
            self.to_state(action)
            self.do_action(response)
            return self.discuss()
        else:
            self.history.append({"role": "assistant", "content": response})
            print(self.history)
            if self.state != "MORE":
                self.reset()
            elif self.state != "EXIT":
                self.reset_to_previous_state()
            return response

    def discuss_from_audio(self, file):
        """
        This function is used to discuss from an audio file.

        Args:
            file: The audio file.

        Returns:
            The response of the conversation.
        """
        if file:
            # Transcribe the audio file and use the input to start the discussion
            return self.discuss(f"{self.transcribe_audio(file)}")
        # Empty output if there is no file
        return ""


if __name__ == "__main__":
    chat = Chat()

    # 마이크 모드
    # gr.Interface(
    #     theme=gr.themes.Soft(),
    #     fn=chat.discuss_from_audio,
    #     live=True,
    #     inputs=gr.Audio(sources="microphone", type="filepath"),
    #     outputs="text",
    # ).launch()

    # 텍스트 모드
    gr.Interface(
        theme=gr.themes.Soft(),
        fn=chat.discuss,
        inputs=gr.Textbox(placeholder="Enter your message here..."),
        outputs="text",
    ).launch()
