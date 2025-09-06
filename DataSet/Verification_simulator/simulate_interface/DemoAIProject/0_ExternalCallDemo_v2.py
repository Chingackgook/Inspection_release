from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.DemoAIProject import *
exe = Executor('DemoAIProject', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/Inspection/Demo/DemoAIProject/external_call_demo.py'
from ai_agent import AIAgent

class ExternalCallDemo:
    """External Call Demo Class - Demonstrates how to use AI agent in external projects"""

    def __init__(self):
        """Initialize external call example"""
        self.ai_agent = None
        self.setup_agent()

    def setup_agent(self):
        """Setup AI agent"""
        try:
            self.ai_agent = exe.create_interface_objects(interface_class_name='AIAgent')
            print('✅ AI agent setup successful')
        except Exception as e:
            print(f'❌ AI agent setup failed: {e}')

    def simple_chat_example(self):
        """Simple chat example"""
        print('\n=== Simple Chat Example ===')
        if not self.ai_agent:
            print('Simulated conversation: Hello! I am an AI assistant.')
            return
        questions = ['Who are you?', 'Please explain what machine learning is', 'Summarize the development prospects of artificial intelligence in one sentence']
        for question in questions:
            print(f'\n📝 Question: {question}')
            result = exe.run('chat', message=question)
            if result['success']:
                print(f'🤖 Answer: {result['reply']}')
                print(f'📊 Tokens used: {result.get('tokens_used', {})}')
            else:
                print(f'❌ Error: {result['error']}')

    def batch_processing_example(self):
        """Batch processing example"""
        print('\n=== Batch Processing Example ===')
        tasks = [{'type': 'translate', 'content': 'Hello, how are you?', 'instruction': 'Translate to Chinese'}, {'type': 'summarize', 'content': 'Artificial intelligence technology is developing rapidly, with breakthroughs in deep learning, natural language processing and other technologies.', 'instruction': 'Summarize in one sentence'}, {'type': 'creative', 'content': 'Spring', 'instruction': 'Write a short poem about spring'}]
        for i, task in enumerate(tasks, 1):
            print(f'\nTask {i}: {task['type']}')
            print(f'Content: {task['content']}')
            if not self.ai_agent:
                print(f"Simulated result: Processed '{task['content']}'")
                continue
            prompt = f'{task['instruction']}: {task['content']}'
            result = exe.run('chat', message=prompt)
            if result['success']:
                print(f'Result: {result['reply']}')
            else:
                print(f'Error: {result['error']}')

    def advanced_usage_example(self):
        """Advanced usage example"""
        print('\n=== Advanced Usage Example ===')
        if not self.ai_agent:
            print('Simulated advanced features demo')
            return
        print('\n1. Creative writing (high temperature):')
        creative_result = exe.run('chat', message='Write the beginning of a short story about robots', temperature=0.9)
        if creative_result['success']:
            print(creative_result['reply'])
        print('\n2. Professional translation (system prompt):')
        translation_result = exe.run('chat', message='Artificial intelligence is transforming our world.', system_prompt='You are a professional English-Chinese translation expert, please provide accurate and natural translations.')
        if translation_result['success']:
            print(translation_result['reply'])
        print('\n3. Sentiment analysis:')
        sentiment_result = exe.run('analyze_text', text="Today's meeting was very successful, the team worked well together!", analysis_type='sentiment')
        if sentiment_result['success']:
            print(sentiment_result['reply'])
demo = ExternalCallDemo()
demo.simple_chat_example()
demo.batch_processing_example()
demo.advanced_usage_example()