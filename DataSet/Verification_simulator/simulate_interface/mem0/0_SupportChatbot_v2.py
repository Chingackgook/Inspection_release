from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.mem0 import *
exe = Executor('mem0','simulation')
FILE_RECORD_PATH = exe.now_record_path
# Import the existing package
import os
from typing import List
from typing import Dict
from mem0 import Memory
from datetime import datetime
import anthropic
from openai import OpenAI
# end


class SupportChatbot:

    def __init__(self):
        self.config = {'llm': {'provider': 'openai', 'config': {'model': 'gpt-4', 'temperature': 0.1, 'max_tokens': 2000}}}
        self.openai_client = OpenAI(base_url="https://sg.uiuiapi.com/v1")
        mem = exe.run('from_config', config_dict=self.config)
        exe.adapter.class1_obj = mem
        self.system_context = "\n        You are a helpful customer support agent. Use the following guidelines:\n        - Be polite and professional\n        - Show empathy for customer issues\n        - Reference past interactions when relevant\n        - Maintain consistent information across conversations\n        - If you're unsure about something, ask for clarification\n        - Keep track of open issues and follow-ups\n        "

    def store_customer_interaction(self, user_id: str, message: str, response: str, metadata: Dict=None):
        """Store customer interaction in memory."""
        if metadata is None:
            metadata = {}
        metadata['timestamp'] = datetime.now().isoformat()
        conversation = [{'role': 'user', 'content': message}, {'role': 'assistant', 'content': response}]
        exe.run('add', messages=conversation, user_id=user_id, metadata=metadata)

    def get_relevant_history(self, user_id: str, query: str) -> List[Dict]:
        """Retrieve relevant past interactions."""
        return exe.run('search', query=query, user_id=user_id, limit=5)

    def handle_customer_query(self, user_id: str, query: str) -> str:
        """Process customer query with context from past interactions."""
        relevant_history = self.get_relevant_history(user_id, query)['results']
        context = 'Previous relevant interactions:\n'
        for memory in relevant_history:
            context += f"Customer: {memory['memory']}\n"
            context += f"Support: {memory['memory']}\n"
            context += '---\n'
        prompt = f'\n        {self.system_context}\n\n        {context}\n\n        Current customer query: {query}\n\n        Provide a helpful response that takes into account any relevant past interactions.\n        '
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.1
        )
        self.store_customer_interaction(
            user_id=user_id,
            message=query,
            response=response.choices[0].message.content,
            metadata={"type": "support_query"}
        )
        return response.choices[0].message.content

# Main logic execution
chatbot = SupportChatbot()
user_id = 'customer_bot'

# Parts that may need manual modification:
queries = ['What are your support hours?', 'I need help with my order.', 'Can you tell me about your return policy?', 'exit']
# end

print('Welcome to Customer Support!')
for query in queries:
    print('Customer:', query)
    if query.lower() == 'exit':
        print('Thank you for using our support service. Goodbye!')
        break
    response = chatbot.handle_customer_query(user_id, query)
    print('Support:', response, '\n\n')
