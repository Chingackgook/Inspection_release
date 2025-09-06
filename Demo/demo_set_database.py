from Inspection.utils.interface_writer import InterfaceWriter
from Inspection.utils.path_manager import INTERFACE_TXT_PATH , DEMO_PATH


DEMO_PROJECT_PATH = DEMO_PATH + 'DemoAIProject/'

# 1: Simulate manual input of interface call-side information into text files
# 1.1: Simulate manual input of call-side code into text files
with open(DEMO_PROJECT_PATH + 'external_call_demo.py', 'r') as f:
    call_code = f.read()
with open(INTERFACE_TXT_PATH + 'APICalls/code.txt', 'w') as f:
    f.write(call_code)
# 1.2: Simulate manual input of call-side name into text files
with open(INTERFACE_TXT_PATH + 'APICalls/name.txt', 'w') as f:
    f.write('ExternalCallDemo')
# 1.3: Simulate manual input of call-side description into text files
with open(INTERFACE_TXT_PATH + 'APICalls/description.txt', 'w') as f:
    f.write('External Call Demo - Demonstrates how to use AI agent in other projects')
# 1.4: Simulate manual input of call location code into text files
with open(INTERFACE_TXT_PATH + 'APICalls/path.txt', 'w') as f:
    f.write(str(DEMO_PROJECT_PATH + 'external_call_demo.py'))


# 2: Simulate manual input of interface implementation-side information into text files
# 2.1: Simulate manual input of interface implementation-side code into text files
with open(DEMO_PROJECT_PATH + 'ai_agent.py', 'r') as f:
    ipl_code = f.read()
with open(INTERFACE_TXT_PATH + 'APIImplementations/implementation.txt', 'w') as f:
    f.write(ipl_code)
# 2.2: Simulate manual input of interface implementation-side name into text files
with open(INTERFACE_TXT_PATH + 'APIImplementations/name.txt', 'w') as f:
    f.write('AIAgent')
# 2.3: Simulate manual input of interface implementation-side description into text files
with open(INTERFACE_TXT_PATH + 'APIImplementations/description.txt', 'w') as f:
    f.write('Intelligent AI agent class for interacting with GPT API')
# 2.4: Simulate manual input of interface implementation-side path into text files
with open(INTERFACE_TXT_PATH + 'APIImplementations/path.txt', 'w') as f:
    f.write(str(DEMO_PROJECT_PATH + 'ai_agent.py'))
# 2.5: Simulate manual input of interface implementation-side call example into text files
with open(INTERFACE_TXT_PATH + 'APIImplementations/example.txt', 'w') as f:
    f.write('\n')

# 3: Write original project root directory
with open(INTERFACE_TXT_PATH + 'ProjectRoot.txt', 'w') as f:
    f.write(str(DEMO_PROJECT_PATH))

# 4: Call interface writing tool class to generate interface information file
iw = InterfaceWriter('DemoAIProject',cover=True)
iw.write()


print("✅ Demo project interface information set completed")

# The generated interface information will be saved in InterfaceData/InterfaceInfo/DemoAIProject.json