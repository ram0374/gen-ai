#https://github.com/pinecone-io/examples/blob/master/learn/generation/langchain/handbook/06-langchain-agents.ipynb

import os
# from getpass import getpass
# OPENAI_API_KEY = getpass()
# Set API keys
os.environ["OPENAI_API_KEY"] = "sk-proj--fQnuUJZ7E8jT3BlbkFJXsS_-66RYdBzRtF_war-Tj0I48kpAsbFu_zPxFgucgOGERa9L7VMtXpM0oTJwHcuyD7QeNadcA"
os.environ["SERPAPI_API_KEY"] = "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"

from langchain import OpenAI

llm = OpenAI(
    model_name="gpt-4o-mini",
    openai_api_key="OPENAI_API_KEY",
    temperature=0
)

from langchain.callbacks import get_openai_callback

def count_tokens(agent, query):
    with get_openai_callback() as cb:
        result = agent(query)
        print(f'Spent a total of {cb.total_tokens} tokens')
    return result