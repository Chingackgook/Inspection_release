import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.gpt_researcher import ENV_DIR
from Inspection.adapters.custom_adapters.gpt_researcher import *
exe = Executor('gpt_researcher', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
from gpt_researcher import GPTResearcher
from backend.utils import write_md_to_pdf
import asyncio

async def main():
    # Load the model
    exe.create_interface_objects(query="What are the most effective ways for beginners to start investing?", report_type="deep")

    # Progress callback
    def on_progress(progress):
        print(f"Depth: {progress.current_depth}/{progress.total_depth}")
        print(f"Breadth: {progress.current_breadth}/{progress.total_breadth}")
        print(f"Queries: {progress.completed_queries}/{progress.total_queries}")
        if progress.current_query:
            print(f"Current query: {progress.current_query}")

    # Set verbose output

    # Run research with progress tracking
    print("Starting deep research...")
    context = await exe.run("conduct_research", on_progress=on_progress)
    print("\nResearch completed. Generating report...")
    
    # Generate the final report
    report = await exe.run("write_report")
    
    # Optionally, write introduction and conclusion
    intro = await exe.run("write_introduction")
    conclusion = await exe.run("write_report_conclusion", report_body=report)

    # Get research costs
    costs = await exe.run("get_costs")
    
    # Write the report to PDF
    await write_md_to_pdf(report, os.path.join(FILE_RECORD_PATH, "deep_research_report.pdf"))
    print(f"\nFinal Report: {report}")
    print(f"Introduction: {intro}")
    print(f"Conclusion: {conclusion}")
    print(f"Research Costs: {costs}")

# Directly run the main logic
asyncio.run(main())
