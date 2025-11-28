#!/usr/bin/env python3
"""
Simple Jira Issue and Log Analysis Script
Fetches Jira issues and extracts error lines from logs without AI models.
"""

import os
import re
import json
import requests
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JiraConfig:
    """Configuration for Jira API access."""
    
    def __init__(self):
        self.base_url = os.getenv('JIRA_BASE_URL', 'https://jira.harman.com/jira')
        self.token = os.getenv('JIRA_BEARER_TOKEN')
        if not self.token:
            raise ValueError("Jira bearer token not configured in JIRA_BEARER_TOKEN")

    def get_auth_header(self) -> Dict[str, str]:
        return {'Authorization': f'Bearer {self.token}'}

class LogProcessor:
    """Processes log files to extract error information."""
    
    def __init__(self):
        self.error_keywords = [
            'ERROR', 'FATAL', 'Exception', 'Traceback', 'CRITICAL', 
            'FAILURE', 'FAILED', 'java.lang.', 'NullPointerException',
            'SQLException', 'RuntimeException', 'IOException'
        ]
    
    def extract_error_context_block(
        self, 
        log_lines: List[str], 
        keywords: Optional[List[str]] = None,
        context_lines: int = 5, 
        max_total_lines: int = 150
    ) -> str:
        """
        Extracts error context from log lines.
        
        Args:
            log_lines: List of log lines
            keywords: Keywords to search for (uses default if None)
            context_lines: Number of context lines before/after error
            max_total_lines: Maximum lines to return
            
        Returns:
            Formatted string with error context
        """
        if keywords is None:
            keywords = self.error_keywords
            
        # Find all line indices that contain error keywords
        error_indices = []
        pattern = re.compile("|".join(re.escape(kw) for kw in keywords), re.IGNORECASE)

        for i, line in enumerate(log_lines):
            if pattern.search(line):
                error_indices.append(i)

        # If no errors found, return last portion of log
        if not error_indices:
            logger.warning("No error keywords found in log, returning last portion")
            start_index = max(0, len(log_lines) - min(max_total_lines, 50))
            return "\n".join([
                "--- [NO SPECIFIC ERRORS FOUND - SHOWING LOG TAIL] ---",
                *log_lines[start_index:],
                "--- [END OF LOG TAIL] ---"
            ])

        # Get error block boundaries
        first_error_index = error_indices[0]
        last_error_index = error_indices[-1]

        # Calculate context boundaries
        start_index = max(0, first_error_index - context_lines)
        end_index = min(len(log_lines), last_error_index + 1 + context_lines)
        
        contextual_block = log_lines[start_index:end_index]

        # Enforce line limit
        if len(contextual_block) > max_total_lines:
            half_max = max_total_lines // 2
            contextual_block = (
                contextual_block[:half_max]
                + ["... [LOG TRUNCATED DUE TO LENGTH] ..."]
                + contextual_block[-half_max:]
            )

        return "\n".join([
            "--- [START OF ERROR CONTEXT] ---",
            *contextual_block,
            "--- [END OF ERROR CONTEXT] ---"
        ])
    
    def analyze_errors(self, log_content: str) -> Dict[str, Any]:
        """
        Analyzes log content and extracts error statistics.
        
        Args:
            log_content: Raw log content
            
        Returns:
            Dictionary with error analysis
        """
        log_lines = log_content.splitlines()
        error_stats = {}
        
        for keyword in self.error_keywords:
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            matches = [line for line in log_lines if pattern.search(line)]
            if matches:
                error_stats[keyword] = {
                    'count': len(matches),
                    'first_occurrence': matches[0][:100] + '...' if len(matches[0]) > 100 else matches[0],
                    'last_occurrence': matches[-1][:100] + '...' if len(matches[-1]) > 100 else matches[-1]
                }
        
        return {
            'total_lines': len(log_lines),
            'error_keywords_found': list(error_stats.keys()),
            'error_statistics': error_stats,
            'error_context': self.extract_error_context_block(log_lines)
        }

class JiraAnalyzer:
    """Main class for Jira issue and log analysis."""
    
    def __init__(self):
        self.config = JiraConfig()
        self.log_processor = LogProcessor()
        self.session = requests.Session()
        self.session.headers.update({
            **self.config.get_auth_header(),
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def fetch_issues(self, project_id: str, days_back: int = 3) -> Dict[str, Any]:
        """
        Fetches Jira issues from specified project.
        
        Args:
            project_id: Jira project ID
            days_back: Number of days to look back
            
        Returns:
            Dictionary with issue data
        """
        try:
            jql = f'project="{project_id}" AND created >= "-{days_back}d"'
            fields = "key,summary,created,status,attachment,description,assignee,reporter"
            url = f"{self.config.base_url}/rest/api/2/search"
            
            params = {
                'jql': jql,
                'fields': fields,
                'maxResults': 50
            }
            
            logger.info(f"Fetching issues for project {project_id} (last {days_back} days)")
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            issues = data.get('issues', [])
            
            result = {
                'total_issues': len(issues),
                'project_id': project_id,
                'query_date': datetime.now().isoformat(),
                'issues': []
            }
            
            for issue in issues:
                fields = issue.get('fields', {})
                attachments = fields.get('attachment', [])
                
                # Filter for log-like attachments
                log_attachments = [
                    att for att in attachments
                    if any(ext in att.get('filename', '').lower() 
                          for ext in ['.log', '.txt', '.out', '.err'])
                ]
                
                issue_data = {
                    'key': issue.get('key'),
                    'summary': fields.get('summary'),
                    'description': fields.get('description'),
                    'created': fields.get('created'),
                    'status': fields.get('status', {}).get('name'),
                    'assignee': fields.get('assignee', {}).get('displayName') if fields.get('assignee') else None,
                    'reporter': fields.get('reporter', {}).get('displayName') if fields.get('reporter') else None,
                    'total_attachments': len(attachments),
                    'log_attachments': [
                        {
                            'id': att.get('id'),
                            'filename': att.get('filename'),
                            'size': att.get('size'),
                            'mimeType': att.get('mimeType')
                        }
                        for att in log_attachments
                    ]
                }
                result['issues'].append(issue_data)
            
            logger.info(f"Successfully fetched {len(issues)} issues")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Jira issues: {e}")
            raise
    
    def download_attachment(self, attachment_id: str, filename: str) -> str:
        """
        Downloads attachment content from Jira.
        
        Args:
            attachment_id: Jira attachment ID
            filename: Attachment filename
            
        Returns:
            Attachment content as string
        """
        try:
            url = f"{self.config.base_url}/secure/attachment/{attachment_id}/{filename}"
            logger.info(f"Downloading attachment: {filename}")
            
            response = self.session.get(url, timeout=120)
            response.raise_for_status()
            
            # Try to decode as text
            try:
                content = response.text
            except UnicodeDecodeError:
                # If it's binary, try to decode with error handling
                content = response.content.decode('utf-8', errors='ignore')
            
            logger.info(f"Successfully downloaded {filename} ({len(content)} characters)")
            return content
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading attachment {filename}: {e}")
            raise
    
    def analyze_issue_logs(self, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes logs for a specific issue.
        
        Args:
            issue_data: Issue data dictionary
            
        Returns:
            Dictionary with log analysis results
        """
        issue_key = issue_data.get('key')
        log_attachments = issue_data.get('log_attachments', [])
        
        if not log_attachments:
            return {
                'issue_key': issue_key,
                'status': 'no_log_attachments',
                'message': 'No log attachments found for this issue'
            }
        
        analysis_results = {
            'issue_key': issue_key,
            'issue_summary': issue_data.get('summary'),
            'log_analyses': []
        }
        
        for attachment in log_attachments:
            try:
                content = self.download_attachment(
                    attachment['id'], 
                    attachment['filename']
                )
                
                log_analysis = self.log_processor.analyze_errors(content)
                log_analysis['attachment_info'] = attachment
                
                analysis_results['log_analyses'].append(log_analysis)
                
            except Exception as e:
                logger.error(f"Error analyzing log {attachment['filename']}: {e}")
                analysis_results['log_analyses'].append({
                    'attachment_info': attachment,
                    'error': str(e),
                    'status': 'failed'
                })
        
        return analysis_results
    
    def run_analysis(self, project_id: str, days_back: int = 3, output_dir: str = 'output') -> Dict[str, Any]:
        """
        Runs complete analysis workflow.
        
        Args:
            project_id: Jira project ID
            days_back: Days to look back
            output_dir: Output directory for results
            
        Returns:
            Complete analysis results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Fetch issues
        logger.info("Starting Jira issue analysis")
        issues_data = self.fetch_issues(project_id, days_back)
        
        # Save issues data
        issues_file = os.path.join(output_dir, f'jira_issues_{project_id}.json')
        with open(issues_file, 'w') as f:
            json.dump(issues_data, f, indent=2, default=str)
        logger.info(f"Issues data saved to: {issues_file}")
        
        # Analyze logs for each issue
        complete_analysis = {
            'project_id': project_id,
            'analysis_date': datetime.now().isoformat(),
            'total_issues': issues_data['total_issues'],
            'issues_with_logs': 0,
            'issue_analyses': []
        }
        
        for issue in issues_data['issues']:
            if issue.get('log_attachments'):
                logger.info(f"Analyzing logs for issue: {issue['key']}")
                log_analysis = self.analyze_issue_logs(issue)
                complete_analysis['issue_analyses'].append(log_analysis)
                complete_analysis['issues_with_logs'] += 1
        
        # Save complete analysis
        analysis_file = os.path.join(output_dir, f'log_analysis_{project_id}.json')
        with open(analysis_file, 'w') as f:
            json.dump(complete_analysis, f, indent=2, default=str)
        logger.info(f"Log analysis saved to: {analysis_file}")
        
        # Generate summary report
        self.generate_summary_report(complete_analysis, output_dir)
        
        return complete_analysis
    
    def generate_summary_report(self, analysis_data: Dict[str, Any], output_dir: str):
        """
        Generates a human-readable summary report.
        
        Args:
            analysis_data: Complete analysis data
            output_dir: Output directory
        """
        report_file = os.path.join(output_dir, f'summary_report_{analysis_data["project_id"]}.md')
        
        with open(report_file, 'w') as f:
            f.write(f"# Jira Log Analysis Summary Report\n\n")
            f.write(f"**Project:** {analysis_data['project_id']}\n")
            f.write(f"**Analysis Date:** {analysis_data['analysis_date']}\n")
            f.write(f"**Total Issues:** {analysis_data['total_issues']}\n")
            f.write(f"**Issues with Logs:** {analysis_data['issues_with_logs']}\n\n")
            
            for issue_analysis in analysis_data['issue_analyses']:
                f.write(f"## Issue: {issue_analysis['issue_key']}\n\n")
                f.write(f"**Summary:** {issue_analysis.get('issue_summary', 'N/A')}\n\n")
                
                for log_analysis in issue_analysis['log_analyses']:
                    if 'error' in log_analysis:
                        f.write(f"### Log: {log_analysis['attachment_info']['filename']} (ERROR)\n")
                        f.write(f"Error: {log_analysis['error']}\n\n")
                        continue
                    
                    f.write(f"### Log: {log_analysis['attachment_info']['filename']}\n\n")
                    f.write(f"- **Total Lines:** {log_analysis['total_lines']}\n")
                    f.write(f"- **Error Keywords Found:** {', '.join(log_analysis['error_keywords_found'])}\n\n")
                    
                    if log_analysis['error_statistics']:
                        f.write("**Error Statistics:**\n")
                        for keyword, stats in log_analysis['error_statistics'].items():
                            f.write(f"- {keyword}: {stats['count']} occurrences\n")
                        f.write("\n")
                    
                    f.write("**Error Context:**\n")
                    f.write("```\n")
                    f.write(log_analysis['error_context'])
                    f.write("\n```\n\n")
                
                f.write("---\n\n")
        
        logger.info(f"Summary report saved to: {report_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Jira Issue and Log Analysis Tool (No AI Models)'
    )
    parser.add_argument(
        '--project-id', 
        required=True, 
        help='Jira project ID (e.g., MSILDA28)'
    )
    parser.add_argument(
        '--days-back', 
        type=int, 
        default=3, 
        help='Number of days to look back for issues (default: 3)'
    )
    parser.add_argument(
        '--output-dir', 
        default='output', 
        help='Output directory for results (default: output)'
    )
    
    args = parser.parse_args()
    
    # Check for required environment variables
    if not os.getenv('JIRA_BEARER_TOKEN'):
        print("ERROR: JIRA_BEARER_TOKEN environment variable is not set!")
        print("Please set it with: export JIRA_BEARER_TOKEN='your_token_here'")
        return 1
    
    try:
        analyzer = JiraAnalyzer()
        results = analyzer.run_analysis(
            project_id=args.project_id,
            days_back=args.days_back,
            output_dir=args.output_dir
        )
        
        print("\n" + "="*60)
        print("JIRA LOG ANALYSIS COMPLETED")
        print("="*60)
        print(f"✓ Project: {results['project_id']}")
        print(f"✓ Total Issues: {results['total_issues']}")
        print(f"✓ Issues with Logs: {results['issues_with_logs']}")
        print(f"✓ Results saved to: {args.output_dir}/")
        print("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())