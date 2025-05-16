#!/usr/bin/env python3

"""
Script to test if your OpenAI API key works for GPT-4o invocations using the LangChain OpenAI library.
Usage:
  1. Install dependencies: pip install langchain openai
  2. Set your API key: export OPENAI_API_KEY="your_api_key"
  3. Run: python test_gpt4o_key.py
"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()
from langchain_openai.chat_models import ChatOpenAI
import openai

def main():
    # Retrieve API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    try:
        # Initialize the LangChain OpenAI LLM for GPT-4o
        llm = ChatOpenAI(
            api_key=api_key,
            model_name="gpt-4o",
            temperature=0
        )

        # Send a simple test prompt
        response = llm.invoke(
            "Hello! Please respond with 'GPT-4o is working' if this invocation succeeds."
        )

        print("\n✅ API Key is valid. Model responded:")
        print(response)

    except Exception as err:
        error_text = str(err)
        # Basic check for authentication-related issues
        if "invalid" in error_text.lower() or "unauthorized" in error_text.lower():
            print("\n❌ Authentication error: API key invalid or unauthorized.")
        else:
            print("\n❌ OpenAI API error occurred.")
        print(error_text)
        sys.exit(1)


if __name__ == "__main__":
    main()

