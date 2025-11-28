#!/usr/bin/env python3
import os
import json
import argparse
import difflib
from pathlib import Path
from typing import List, Dict, Type

import faiss
import numpy as np
import yaml
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer

# --- Load Environment Variables ---
load_dotenv()

# --- Argument Parser ---
parser = argparse.ArgumentParser(
    description="AI agent to analyze an RCA with multiple errors and produce a consolidated code fix report."
)
parser.add_argument("--rca_file", required=True, help="Path to the RCA markdown file.")
parser.add_argument("--faiss_dir", default="faiss_index", help="Directory of the FAISS index.")
parser.add_argument("--code_dir", default="downloaded_files", help="Directory of the source code.")
parser.add_argument("--output_file", default="iterative_code_fix_report.md", help="Path to save the final report.")
parser.add_argument("--top_k", default=3, type=int, help="Number of top relevant chunks to retrieve.")
args = parser.parse_args()


# --- Custom Tool Definitions ---

class FaissChunkSearchTool(BaseTool):
    """A tool to find relevant code chunks from a FAISS index based on a query."""
    name: str = "FAISS Code Chunk Search"
    description: str = (
        "Searches a FAISS index to find the most relevant code chunks for a given problem description. "
        "Returns multiple relevant code chunks with their context. "
        "Input must be a concise, natural language query about the error or bug."
    )
    _model: SentenceTransformer = None
    _index: faiss.Index = None
    _metadata: List[Dict] = None
    _top_k: int = 3

    def __init__(self, faiss_dir: Path, top_k: int = 3):
        super().__init__()
        self._top_k = top_k
        metadata_path = faiss_dir / "metadata.json"
        index_path = faiss_dir / "code_index.faiss"
        if not all([metadata_path.exists(), index_path.exists()]):
            raise FileNotFoundError(f"FAISS index or metadata not found in {faiss_dir}.")

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_content = json.load(f)
            self._metadata = metadata_content["chunks"]
            model_name = metadata_content.get('model_name', 'all-MiniLM-L6-v2')
        
        self._model = SentenceTransformer(model_name)
        self._index = faiss.read_index(str(index_path))

    def _run(self, query: str) -> str:
        """Executes the search and returns the most relevant code chunks."""
        query_embedding = self._model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        scores, indices = self._index.search(query_embedding, self._top_k)

        if not indices.size:
            return "Error: No relevant code chunks found."
        
        results = []
        seen_files = set()
        
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # Invalid index
                continue
                
            chunk_info = self._metadata[idx]
            filename = chunk_info['filename']
            chunk_content = chunk_info.get('content', chunk_info.get('chunk', ''))
            start_line = chunk_info.get('start_line', 0)
            end_line = chunk_info.get('end_line', 0)
            
            # Add file context if we haven't seen this file before
            file_context = ""
            if filename not in seen_files:
                file_context = f"\n--- FILE: {filename} ---\n"
                seen_files.add(filename)
            
            results.append(f"{file_context}Lines {start_line}-{end_line} (Relevance: {score:.3f}):\n```\n{chunk_content}\n```\n")
        
        if not results:
            return "Error: No valid code chunks found."
            
        return "RELEVANT CODE CHUNKS:\n" + "\n".join(results)

class FileContextTool(BaseTool):
    """A tool to get additional context around specific lines in a file."""
    name: str = "File Context Reader"
    description: str = (
        "Reads additional context around specific lines in a file. "
        "Input format: 'filename:start_line:end_line:context_lines' where context_lines is optional (default 10)."
    )
    _code_dir: Path

    def __init__(self, code_dir: Path):
        super().__init__()
        self._code_dir = code_dir

    def _run(self, input_str: str) -> str:
        """Reads and returns the content around specific lines with context."""
        try:
            parts = input_str.split(':')
            if len(parts) < 3:
                return "Error: Input format should be 'filename:start_line:end_line:context_lines'"
            
            filename = parts[0]
            start_line = int(parts[1])
            end_line = int(parts[2])
            context_lines = int(parts[3]) if len(parts) > 3 else 10
            
            safe_path = self._code_dir.joinpath(filename).resolve()
            if not safe_path.is_relative_to(self._code_dir.resolve()):
                return f"Error: Access to path '{filename}' is not allowed."

            if not safe_path.exists():
                return f"Error: File not found at '{filename}'."
            
            with open(safe_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Calculate context bounds
            context_start = max(0, start_line - 1 - context_lines)
            context_end = min(len(lines), end_line + context_lines)
            
            result_lines = []
            for i in range(context_start, context_end):
                line_num = i + 1
                prefix = ">>> " if start_line <= line_num <= end_line else "    "
                result_lines.append(f"{prefix}{line_num:4d}: {lines[i].rstrip()}")
            
            return f"CONTEXT FOR {filename} (lines {context_start+1}-{context_end}):\n" + "\n".join(result_lines)
            
        except (ValueError, IndexError) as e:
            return f"Error parsing input: {str(e)}"

class CompleteFileReaderTool(BaseTool):
    """A tool to read complete files when needed for comprehensive fixes."""
    name: str = "Complete File Reader"
    description: str = (
        "Reads the complete content of a file. Use this sparingly, only when you need to see "
        "the entire file structure for comprehensive fixes. Input must be a valid file path."
    )
    _code_dir: Path

    def __init__(self, code_dir: Path):
        super().__init__()
        self._code_dir = code_dir

    def _run(self, filepath: str) -> str:
        """Reads and returns the complete content of the file."""
        safe_path = self._code_dir.joinpath(filepath).resolve()
        if not safe_path.is_relative_to(self._code_dir.resolve()):
            return f"Error: Access to path '{filepath}' is not allowed."

        if not safe_path.exists():
            return f"Error: File not found at '{filepath}'."
        
        try:
            with open(safe_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add line numbers for easier reference
            lines = content.split('\n')
            numbered_lines = [f"{i+1:4d}: {line}" for i, line in enumerate(lines)]
            return f"COMPLETE FILE: {filepath}\n" + "\n".join(numbered_lines)
        except Exception as e:
            return f"Error reading file: {str(e)}"

# --- Agent and Task Definitions ---

llm = LLM(
    model="azure/gpt-4o",
    api_version="2025-01-01-preview",
    temperature=0.1,  # Lower temperature for more consistent output
)

# Initialize tools
faiss_tool = FaissChunkSearchTool(faiss_dir=Path(args.faiss_dir), top_k=args.top_k)
context_tool = FileContextTool(code_dir=Path(args.code_dir))
complete_file_tool = CompleteFileReaderTool(code_dir=Path(args.code_dir))

# Enhanced agent with better instructions
code_fixer_agent = Agent(
    role="Autonomous Code Chunk Analyzer and Fixer",
    goal=(
        "Analyze an RCA file to identify all described errors. For each error, find the relevant code chunks, "
        "analyze the specific problematic sections, generate targeted fixes, and compile all fixes into a "
        "comprehensive markdown report. Focus on chunk-level analysis rather than full file processing."
    ),
    backstory=(
        "You are an expert AI developer specializing in precise code analysis and targeted fixes. You excel at "
        "identifying problematic code sections, understanding their context within larger codebases, and creating "
        "minimal but effective fixes. You always complete your reports fully and systematically work through "
        "each problem until all are addressed."
    ),
    tools=[faiss_tool, context_tool, complete_file_tool],
    llm=llm,
    max_iter=30,  # Increased iteration limit
      # 30 minutes timeout
    verbose=True,
    allow_delegation=False,  # Prevent delegation to ensure completion
)

# Enhanced task with more specific instructions
iterative_fix_task = Task(
    description=f"""
CRITICAL: You MUST complete this entire task and provide a full consolidated report. Do not stop until you have addressed ALL problems found in the RCA.

**RCA File Content:**
---
{Path(args.rca_file).read_text()}
---

**MANDATORY PROCESS - Complete ALL steps:**

1. **Parse RCA Thoroughly:** 
   - Read the RCA file completely
   - Identify and list ALL distinct problems/errors described
   - Number each problem clearly (Problem 1, Problem 2, etc.)

2. **For EACH Problem (do not skip any):**
   a. Use 'FAISS Code Chunk Search' to find relevant code chunks for the specific problem
   b. Analyze the returned chunks to understand the issue
   c. If you need more context around the problematic code, use 'File Context Reader'
   d. Only use 'Complete File Reader' if the chunk-level analysis is insufficient
   e. Generate a targeted fix for the specific problematic code section
   f. Create before/after code blocks showing the exact changes

3. **Generate Complete Final Report:**
   - Create a comprehensive markdown report with:
     - Main title: "# Code Fix Report"
     - Executive summary of all problems found
     - Section for each problem with:
       - Problem description
       - Affected file(s) and line ranges
       - Root cause analysis
       - Proposed fix with unified diff
       - Before/After code blocks
   - Include a summary table of all fixes at the end

**CRITICAL SUCCESS CRITERIA:**
- You MUST address every single problem mentioned in the RCA
- Your report MUST be complete and well-formatted
- You MUST provide actual code fixes, not just descriptions
- You MUST include unified diffs for each fix
- You MUST NOT stop until the complete report is generated

**Final Output Format:**
Return ONLY the complete markdown content. Start with "# Code Fix Report" and end with a fixes summary table.
""",
    expected_output=(
        "A complete markdown report containing analysis and fixes for ALL problems identified in the RCA. "
        "Must include executive summary, detailed sections for each fix with diffs and code blocks, "
        "and a summary table. The report must be fully complete and ready for use.DONT INCLUDE YOUR THOUGHT IN THE REPORT"
    ),
    agent=code_fixer_agent,
    output_file=args.output_file,
)

# --- Enhanced Crew Configuration ---
crew = Crew(
    agents=[code_fixer_agent],
    tasks=[iterative_fix_task],
    process=Process.sequential,
    verbose=True,
    max_rpm=100,  # Increase rate limit
    share_crew=False,  # Prevent sharing to ensure completion
)

# --- Execution with Error Handling ---
def main():
    print("🚀 Starting Enhanced Chunk-based Code Fix Agent...")
    print(f"📁 RCA File: {args.rca_file}")
    print(f"🔍 FAISS Index: {args.faiss_dir}")  
    print(f"📂 Code Directory: {args.code_dir}")
    print(f"📝 Output File: {args.output_file}")
    print(f"🔢 Top-K Chunks: {args.top_k}")
    print("-" * 60)
    
    try:
        result = crew.kickoff()
        print("✅ Agent execution completed successfully.")
        print(f"📋 Report saved to: {args.output_file}")
        
        # Verify the output file was created and has content
        output_path = Path(args.output_file)
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"✅ Output file verified: {output_path.stat().st_size} bytes")
        else:
            print("⚠️ Warning: Output file is empty or not created")
            
        print("\n" + "="*60)
        print("FINAL REPORT PREVIEW:")
        print("="*60)
        print(result.raw[:1000] + "..." if len(result.raw) > 1000 else result.raw)
        
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        print("🔄 Check the verbose output above for details")
        raise

if __name__ == "__main__":
    main()