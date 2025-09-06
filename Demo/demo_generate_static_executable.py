from Inspection.generator.simulation_generator_v2 import SimulationGenerator
from Inspection.ai.base_ai import BaseAI

# This file corresponds to step 2 of the approach in the paper

# 1: Call the static executable artifact generation tool class to generate static executable files
ai = BaseAI('demo_chat')
sim_gen = SimulationGenerator('DemoAIProject')
sim_gen.set_base_ai(ai.copy())
sim_gen.generate_simulation(call_idx=0) # 0 means generating static executable artifact based on the first interface call code in the interface information

# The generated static executable file will be saved in Inspection/simulation/simulate_interface/DemoAIProject/0_ExternalCallDemo_v2.py