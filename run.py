# interact with chatbot through console

from llm import ChatEng

if __name__ == "__main__":
    system_prompt = ("I am a citizen of Newport News, Virginia asking about the city code. "
                     "You are a Newport News chatbot meant to assist the user. "
                     "Using only the Context provided, answer and explain the users most recent query"
                     )
    chat_bot = ChatEng()
    chat_bot.ask_console(chat_bot)
