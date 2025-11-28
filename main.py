#!/usr/bin/env python3
import gradio as gr
import os
import subprocess
import threading
import time
from pathlib import Path
from dotenv import load_dotenv
import queue
import sys

class PipelineRunner:
    def __init__(self):
        self.current_process = None
        self.output_queue = queue.Queue()
        self.is_running = False
        
    def check_env_vars(self):
        """Checks for the presence of required environment variables."""
        load_dotenv()
        required_vars = [
            "JIRA_BEARER_TOKEN",
            "GITHUB_TOKEN",
        ]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        return missing_vars
    
    def run_pipeline(self, project_id, days_back, repo_url, progress=gr.Progress()):
        """Run the pipeline with progress tracking."""
        self.is_running = True
        
        # Set UTF-8 encoding for subprocess environments
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        # Check environment variables first
        missing_vars = self.check_env_vars()
        if missing_vars:
            error_msg = f"❌ Missing required environment variables: {', '.join(missing_vars)}\nPlease create a .env file or set them in your environment."
            self.is_running = False
            return error_msg, "", ""
        
        # Create output directories
        base_output_dir = Path("pipeline_output")
        jira_output_dir = base_output_dir / "1_jira_analysis"
        rca_output_dir = base_output_dir / "2_rca_report"
        code_download_dir = base_output_dir / "3_downloaded_code"
        faiss_index_dir = base_output_dir / "4_faiss_index"
        final_report_dir = base_output_dir / "5_final_fix_report"

        for d in [jira_output_dir, rca_output_dir, code_download_dir, faiss_index_dir, final_report_dir]:
            d.mkdir(parents=True, exist_ok=True)

        summary_report_file = jira_output_dir / f"summary_report_{project_id}.md"
        rca_analysis_file = rca_output_dir / "rca_analysis.md"
        final_fix_report_file = final_report_dir / "consolidated_code_fix_report.md"
        
        output_log = []
        
        # Pipeline steps
        steps = [
            {
                "name": "Jira Issue and Log Analysis",
                "command": [
                    "python", "JIRA_issue_log_analysis.py",
                    "--project-id", project_id,
                    "--days-back", str(days_back),
                    "--output-dir", str(jira_output_dir)
                ]
            },
            {
                "name": "Root Cause Analysis (RCA) Generation",
                "command": [
                    "python", "rca_bot.py",
                    "--input-file", str(summary_report_file),
                    "--output-file", str(rca_analysis_file)
                ]
            },
            {
                "name": "GitHub File Fetcher",
                "command": [
                    "python", "github_fetcher.py",
                    "--repo_url", repo_url,
                    "--rca_file", str(rca_analysis_file),
                    "--output_dir", str(code_download_dir)
                ]
            },
            {
                "name": "FAISS Code Indexing",
                "command": [
                    "python", "faiss_code_file_indexer.py",
                    "--input_dir", str(code_download_dir),
                    "--output_dir", str(faiss_index_dir),
                    "--rebuild"
                ]
            },
            {
                "name": "Code Fix Generation",
                "command": [
                    "python", "code_fix_ai.py",
                    "--rca_file", str(rca_analysis_file),
                    "--faiss_dir", str(faiss_index_dir),
                    "--code_dir", str(code_download_dir),
                    "--output_file", str(final_fix_report_file)
                ]
            }
        ]
        
        try:
            output_log.append("🚀🚀🚀 Starting Automated Code Fix Pipeline 🚀🚀🚀\n")
            
            for i, step in enumerate(steps):
                if not self.is_running:
                    output_log.append("\n❌ Pipeline stopped by user.\n")
                    break
                    
                progress((i + 1) / len(steps), f"Running: {step['name']}")
                
                output_log.append(f"\n{'='*60}")
                output_log.append(f"🚀 STEP {i+1}/{len(steps)}: {step['name']}")
                output_log.append(f"   Command: {' '.join(step['command'])}")
                output_log.append(f"{'='*60}\n")
                
                try:
                    process = subprocess.Popen(
                        step['command'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True,
                        env=env,
                        encoding='utf-8',
                        errors='replace'  # Replace problematic characters instead of failing
                    )
                    
                    self.current_process = process
                    
                    # Read output in real-time
                    while True:
                        output = process.stdout.readline()
                        if output == '' and process.poll() is not None:
                            break
                        if output:
                            output_log.append(output.strip())
                    
                    return_code = process.poll()
                    
                    if return_code != 0:
                        output_log.append(f"\n❌ ERROR: {step['name']} failed with exit code {return_code}")
                        self.is_running = False
                        return "\n".join(output_log), "", ""
                    else:
                        output_log.append(f"\n✅ COMPLETED: {step['name']}")
                        
                except FileNotFoundError:
                    output_log.append(f"\n❌ ERROR: Command not found for '{step['name']}'. Is '{step['command'][0]}' in your PATH?")
                    self.is_running = False
                    return "\n".join(output_log), "", ""
                except Exception as e:
                    output_log.append(f"\n❌ An unexpected error occurred during '{step['name']}': {e}")
                    self.is_running = False
                    return "\n".join(output_log), "", ""
            
            if self.is_running:
                output_log.append(f"\n{'='*60}")
                output_log.append("🎉🎉🎉 Pipeline Completed Successfully! 🎉🎉🎉")
                output_log.append(f"Final report location: {final_fix_report_file}")
                output_log.append(f"{'='*60}")
                
                # Try to read the final report
                final_report_content = ""
                rca_content = ""
                
                try:
                    if final_fix_report_file.exists():
                        with open(final_fix_report_file, 'r', encoding='utf-8', errors='replace') as f:
                            final_report_content = f.read()
                except Exception as e:
                    final_report_content = f"Error reading final report: {e}"
                
                try:
                    if rca_analysis_file.exists():
                        with open(rca_analysis_file, 'r', encoding='utf-8', errors='replace') as f:
                            rca_content = f.read()
                except Exception as e:
                    rca_content = f"Error reading RCA report: {e}"
                
                self.is_running = False
                return "\n".join(output_log), rca_content, final_report_content
            
        except Exception as e:
            output_log.append(f"\n❌ Pipeline failed with error: {e}")
            self.is_running = False
            return "\n".join(output_log), "", ""
        
        self.is_running = False
        return "\n".join(output_log), "", ""
    
    def stop_pipeline(self):
        """Stop the currently running pipeline."""
        self.is_running = False
        if self.current_process:
            try:
                self.current_process.terminate()
                self.current_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.current_process.kill()
            except Exception:
                pass
        return "Pipeline stop requested..."

# Initialize the pipeline runner
pipeline_runner = PipelineRunner()

def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="Automated Code Fix Pipeline", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚀 Automated Code Fix Pipeline")
        gr.Markdown("This tool runs an automated pipeline to analyze Jira issues, perform root cause analysis, and generate code fixes.")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📋 Configuration")
                
                project_id = gr.Textbox(
                    label="Jira Project ID",
                    placeholder="e.g., MSILDA28",
                    info="The Jira project identifier"
                )
                
                days_back = gr.Number(
                    label="Days Back",
                    value=3,
                    minimum=1,
                    maximum=30,
                    info="Number of days to look back for Jira issues"
                )
                
                repo_url = gr.Textbox(
                    label="GitHub Repository URL",
                    placeholder="https://github.com/owner/repo",
                    info="Full GitHub repository URL"
                )
                
                with gr.Row():
                    run_btn = gr.Button("▶️ Run Pipeline", variant="primary", size="lg")
                    stop_btn = gr.Button("⏹️ Stop Pipeline", variant="stop", size="lg")
                
                gr.Markdown("## 📋 Environment Variables Required")
                gr.Markdown("Make sure these are set in your `.env` file or environment:")
                gr.Markdown("- `JIRA_BEARER_TOKEN`\n- `GITHUB_TOKEN`")
            
            with gr.Column(scale=2):
                gr.Markdown("## 📊 Pipeline Progress & Output")
                
                with gr.Tabs():
                    with gr.TabItem("📈 Live Output"):
                        output_display = gr.Textbox(
                            label="Pipeline Output",
                            lines=25,
                            max_lines=25,
                            show_copy_button=True,
                            interactive=False
                        )
                    
                    with gr.TabItem("🔍 RCA Analysis"):
                        rca_display = gr.Textbox(
                            label="Root Cause Analysis Report",
                            lines=25,
                            max_lines=25,
                            show_copy_button=True,
                            interactive=False
                        )
                    
                    with gr.TabItem("🛠️ Code Fix Report"):
                        fix_display = gr.Textbox(
                            label="Final Code Fix Report",
                            lines=25,
                            max_lines=25,
                            show_copy_button=True,
                            interactive=False
                        )
        
        # Event handlers
        run_btn.click(
            fn=pipeline_runner.run_pipeline,
            inputs=[project_id, days_back, repo_url],
            outputs=[output_display, rca_display, fix_display],
            show_progress=True
        )
        
        stop_btn.click(
            fn=pipeline_runner.stop_pipeline,
            outputs=[output_display]
        )
        
        # Add some example values
        gr.Examples(
            examples=[
                ["MSILDA28", 3, "https://github.com/microsoft/example-repo"],
                ["PROJ123", 7, "https://github.com/company/project"],
            ],
            inputs=[project_id, days_back, repo_url],
            label="Example Configurations"
        )
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )