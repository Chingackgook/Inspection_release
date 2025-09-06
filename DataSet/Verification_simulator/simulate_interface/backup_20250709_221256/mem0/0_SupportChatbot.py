import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.mem0 import ENV_DIR
from Inspection.adapters.custom_adapters.mem0 import *
exe = Executor('mem0','simulation')
FILE_RECORD_PATH = exe.now_record_path


import os
from typing import List, Dict
from mem0 import Memory
from datetime import datetime
import anthropic
from openai import OpenAI

os.environ["OPENAI_BASE_URL"]=os.getenv("OPENAI_API_BASE")

openai_client = OpenAI(base_url="https://sg.uiuiapi.com/v1")
memory = Memory()

class SupportChatbot:
    def __init__(self):
        # Initialize Mem0 with OpenAI's GPT
        self.config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                    "temperature": 0.1,
                    "max_tokens": 2000,
                }
            }
        }
        self.client = openai_client  # Use OpenAI client
        self.memory = exe.run("from_config", config_dict=self.config)

        # Define support context
        self.system_context = """
        You are a helpful customer support agent. Use the following guidelines:
        - Be polite and professional
        - Show empathy for customer issues
        - Reference past interactions when relevant
        - Maintain consistent information across conversations
        - If you're unsure about something, ask for clarification
        - Keep track of open issues and follow-ups
        """

    def store_customer_interaction(self,
                                   user_id: str,
                                   message: str,
                                   response: str,
                                   metadata: Dict = None):
        """Store customer interaction in memory."""
        if metadata is None:
            metadata = {}

        # Add timestamp to metadata
        metadata["timestamp"] = datetime.now().isoformat()

        # Format conversation for storage
        conversation = [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response}
        ]

        # Store in Mem0
        exe.run("add", messages=conversation, user_id=user_id, metadata=metadata)

    def get_relevant_history(self, user_id: str, query: str) -> List[Dict]:
        """Retrieve relevant past interactions."""
        return exe.run("search", query=query, user_id=user_id, limit=5)

    def handle_customer_query(self, user_id: str, query: str) -> str:
        """Process customer query with context from past interactions."""

        # Get relevant past interactions
        relevant_history = self.get_relevant_history(user_id, query)["results"]

        print(f"{relevant_history}")
        # Build context from relevant history
        context = "Previous relevant interactions:\n"
        for memory in relevant_history:
            if type(memory) == str:
                context += f"Customer: {memory}\n"
            else:
                memory_str = memory.get("memory", "")
                context += f"Customer: {memory_str}\n"
            context += "---\n"


        print(context)
        # Prepare prompt with context and current query
        prompt = f"""
        {self.system_context}

        {context}

        Current customer query: {query}

        Provide a helpful response that takes into account any relevant past interactions.
        """

        # Generate response using OpenAI
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.1
        )

        # Store interaction
        self.store_customer_interaction(
            user_id=user_id,
            message=query,
            response=response.choices[0].message.content,
            metadata={"type": "support_query"}
        )

        return response.choices[0].message.content
chatbot = SupportChatbot()
user_id = "customer_bot"
print("Welcome to Customer Support! Type 'exit' to end the conversation.")

    # Get user input
query = "How can I reset my password?,idont know how to reset my password, please help me"
print("Customer:", query)

# Check if user wants to exit

# Handle the query and print the response
response = chatbot.handle_customer_query(user_id, query)
print("Support:", response, "\n\n")