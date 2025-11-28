#!/usr/bin/env python3
import os
import re
import argparse
from crewai import Agent, Task, Crew, LLM

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Generate RCA from a Jira summary report.")
parser.add_argument("--input-file", required=True, help="Path to the summary report markdown file.")
parser.add_argument("--output-file", required=True, help="Path to save the RCA analysis report.")
args = parser.parse_args()

# --- Load and parse the Markdown file ---
try:
    with open(args.input_file, "r", encoding="utf-8") as f:
        content = f.read()
except FileNotFoundError:
    print(f"Error: Input file not found at {args.input_file}")
    exit(1)

summary_match = re.search(r"\*\*Summary:\*\* (.+)", content)
issue_description = summary_match.group(1).strip() if summary_match else "No summary found."

log_contexts = re.findall(r"\*\*Error Context:\*\*\n```(?:\w*\n)?(.*?)```", content, re.DOTALL)
logs_combined = "\n\n".join(log_contexts)

if not logs_combined:
    print("Warning: No error contexts found in the input file. The analysis may be limited.")
    logs_combined = "No logs provided."

# --- Define agent using the Azure LLM ---
rca_agent = Agent(
    role="Root Cause Analysis Expert",
    goal="Analyze Jira issues and logs to produce a detailed root cause analysis. Your report MUST identify the specific filenames believed to be the cause.",
    backstory="You're an experienced software troubleshooter known for uncovering the root of tricky bugs and skilled in troubleshooting by correlating logs with system events.",
    allow_delegation=False,
    llm = LLM(
    model="azure/gpt-4o",
    api_version="2025-01-01-preview"
)
)

# --- Define the RCA task ---
rca_task = Task(
    description=(
        f"### Jira Issue Summary:\n{issue_description}\n\n"
        f"### Combined Log Context:\n{logs_combined}\n\n"
        "**Perform a detailed Root Cause Analysis (RCA) based on the provided Jira issue summary and combined log context.**\n\n"
        "Your analysis should clearly explain the failure, identify the root cause, and propose actionable resolutions. "
        "**Crucially, you must identify and explicitly list the filenames (e.g., `user_service.py`) that are the most likely source of the error.**"
    ),
    expected_output=(
        "A detailed root cause analysis report in markdown, following this strict format:\n\n"
        "## Root Cause Analysis Report\n\n"
        "### 1. Issue Summary\n"
        "*[Briefly restate the problem as described in the Jira issue.]*\n\n"
        "### 2. Failure Explanation\n"
        "*[Provide a clear, concise explanation of what failed and how it manifested, based on the logs and issue description.]*\n\n"
        "### 3. Root Cause\n"
        "*[Detail the underlying cause(s) of the failure. This should be the core of your analysis, explaining 'why' the failure occurred.]*\n\n"
        "### 4. Affected Files\n"
        "*[**List the specific filenames (e.g., `api_handler.js`, `database_schema.sql`)that are identified as the most likely source of the error.**]\n\n"
        "### 5. Suggested Resolutions\n"
        "*[Propose concrete, actionable steps to resolve the issue and prevent recurrence. Prioritize short-term fixes and long-term improvements.]*"
    ),
    agent=rca_agent,
    output_file=args.output_file
)

# --- Execute with Crew ---
crew = Crew(
    agents=[rca_agent],
    tasks=[rca_task],
    verbose=True
)

print(f"🚀 Starting RCA Generation for {args.input_file}...")
result = crew.kickoff()
print("\n✅ RCA Generation Finished.")
print(f"Report saved to: {args.output_file}")
print("\n===== RCA RAW RESULT =====\n")
print(result)