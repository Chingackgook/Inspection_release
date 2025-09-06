from Inspection.generator.dumb_func_generator_v2 import DumbFuncGenerator
from Inspection.ai.base_ai import BaseAI


# This file corresponds to step 3 of the approach in the paper

# 1: Call the non-intelligent module simulation function generation tool class to generate non-intelligent module simulation functions

ai = BaseAI('demo_chat')
dg = DumbFuncGenerator('DemoAIProject')
dg.set_base_ai(ai.copy())
dg.generate_dumb_simulator_function('chat', call_idx=0)

# The generated non-intelligent module simulation function will be saved in Inspection/simulation/dumb_simulator/DemoAIProject/chat_call0_dumbV2.py