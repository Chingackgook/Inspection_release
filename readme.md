# LLM-CRNIMS: An Executable Context Reconstruction and Simulation Framework for Intelligent Software

This repository contains the implementation of **LLM-CRNIMS**, a framework for executable context reconstruction and simulation in Intelligent software.

## Preparation

### System Requirements
- Linux or macOS (Windows will provide support in the future)
- Python 3.8 or higher

### Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure the API settings in `Inspection/config.json`:
   ```json
   {
       "provider": "OpenAI",
       "api_key": "your_api_key_here",
       "base_url": "https://api.openai.com/v1",
       "model": "gpt-4o-mini"
   }
   ```

## Quick Start

To enter the interactive environment, run:

```bash
python -m Inspection
```

## Example Workflow

To reproduce the steps described in the "Approach" section of the paper, execute the following commands in order:

```bash
python -m Demo.demo_set_database
python -m Demo.demo_generate_doc_adapter
python -m Demo.demo_generate_SPE
python -m Demo.demo_generate_NIMASM
python -m Demo.demo_execute_and_suggest
```
