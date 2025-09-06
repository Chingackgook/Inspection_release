from Demo import run_python_module
    
# Execute static executable artifact
module_name = 'Inspection.simulation.simulate_interface.DemoAIProject.0_ExternalCallDemo_v2'
run_python_module(module_name)
# Execution results will be recorded in Records/simulation directory

# Execute parameter simulator
import importlib
module = importlib.import_module('Inspection.simulation.dumb_simulator.DemoAIProject.chat_call0_dumbV2')
dumb_simulator = getattr(module, 'dumb_simulator')
print("=== Obtained simulation parameters for intelligent module ===")
print(dumb_simulator())


# Automatically generate inspection suggestions
from Inspection.ai.base_ai import BaseAI
from Inspection.generator.suggestion_generator import SuggestionGenerator
sg = SuggestionGenerator('DemoAIProject')
sg.set_base_ai(BaseAI('demo_chat'))
sg.generate_suggestions(api_name='chat',idx = 0, simulate_type='simulation')

# Generated inspection suggestions are in Suggestions/ directory


