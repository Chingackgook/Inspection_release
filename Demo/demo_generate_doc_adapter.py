from Inspection.generator.doc_generator import DocGenerator
from Inspection.generator.adapter_generator import AdapterGenerator
from Inspection.ai.base_ai import BaseAI

ai = BaseAI('demo_chat')
# This file corresponds to step 1 of the approach in the paper

# 1: Call the interface documentation generation tool class to generate interface documentation
print("=== Generating Interface Documentation ===")
dg = DocGenerator('DemoAIProject')
dg.set_base_ai(ai.copy())
dg.generate_doc()

# 2: Call the adapter generation tool class to generate adapter code
print("=== Generating Adapter Code ===")
ag = AdapterGenerator('DemoAIProject')
ag.set_base_ai(ai.copy())
ag.generate_adapter()

# The generated interface documentation will be saved in InterfaceData/InterfaceDocs/DemoAIProject.md
# The generated adapter code will be saved in Inspection/adapters/custom_adapters/DemoAIProject.py