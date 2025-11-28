import re
import requests
import base64
import argparse
import time
from urllib.parse import urlparse
from dotenv import load_dotenv
import os
from pathlib import Path

# --- Load token from .env ---
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise EnvironmentError("❌ GitHub token not found in .env file (GITHUB_TOKEN)")

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Download affected files from GitHub repo based on RCA file.")
parser.add_argument("--repo_url", required=True, help="GitHub repository URL (e.g. https://github.com/org/repo)")
parser.add_argument("--rca_file", required=True, help="Path to RCA markdown file (e.g. rca_analysis.md)")
parser.add_argument("--branch", default="main", help="Branch name (default: main)")
parser.add_argument("--output_dir", default="downloaded_files", help="Output directory for downloaded files")
parser.add_argument("--pattern", default=r"\*\*`(.+?\.java)`\*\*", help="Regex pattern to extract filenames")

args = parser.parse_args()

# Create output directory
Path(args.output_dir).mkdir(exist_ok=True)

# --- Extract owner and repo ---
def parse_github_url(repo_url):
    parsed = urlparse(repo_url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError("Invalid GitHub repo URL")
    # Handle .git suffix
    repo_name = parts[1].replace('.git', '')
    return parts[0], repo_name

owner, repo = parse_github_url(args.repo_url)
print(f"🔍 Searching in repository: {owner}/{repo}")

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# --- Step 1: Enhanced file name parsing ---
def extract_file_names(rca_file, pattern):
    """Extract file names using multiple patterns and methods"""
    with open(rca_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Primary pattern (from args)
    file_names = set(re.findall(pattern, content))
    
    # Additional patterns to catch variations
    additional_patterns = [
        r"`([^`]+\.java)`",  # Any .java file in backticks
        r"([A-Za-z][A-Za-z0-9_]*\.java)",  # Simple filename pattern
        r"\b([A-Za-z][A-Za-z0-9_/]*\.java)\b",  # With possible path
        r"File:\s*([^\s]+\.java)",  # "File: filename.java"
        r"- ([A-Za-z][A-Za-z0-9_/]*\.java)",  # List items
    ]
    
    for pattern in additional_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        file_names.update(matches)
    
    # Clean up file names (remove paths, keep just filename)
    cleaned_names = set()
    for name in file_names:
        # Extract just the filename if it contains path separators
        clean_name = name.split('/')[-1].split('\\')[-1]
        if clean_name.endswith('.java'):
            cleaned_names.add(clean_name)
    
    return list(cleaned_names)

file_names = extract_file_names(args.rca_file, args.pattern)
print(f"📄 Found affected files: {file_names}")

# --- Step 2: Enhanced search with multiple strategies ---
def search_file_multiple_strategies(filename):
    """Try multiple search strategies to find the file"""
    strategies = [
        # Strategy 1: Exact filename search
        f"repo:{owner}/{repo} filename:{filename}",
        # Strategy 2: Search in file content (less precise but broader)
        f"repo:{owner}/{repo} {filename}",
        # Strategy 3: Search with file extension
        f"repo:{owner}/{repo} {filename.replace('.java', '')} extension:java",
        # Strategy 4: Search with wildcards (if filename has common prefixes)
        f"repo:{owner}/{repo} filename:{filename.split('.')[0]}*.java",
    ]
    
    for i, query in enumerate(strategies, 1):
        print(f"  🔍 Strategy {i}: {query}")
        try:
            search_url = f"https://api.github.com/search/code"
            params = {"q": query, "per_page": 10}
            
            r = requests.get(search_url, headers=HEADERS, params=params)
            
            if r.status_code == 403:
                print(f"  ⚠️  Rate limited, waiting 60 seconds...")
                time.sleep(60)
                r = requests.get(search_url, headers=HEADERS, params=params)
            
            if r.status_code == 200:
                items = r.json().get("items", [])
                if items:
                    # Sort by relevance (exact filename matches first)
                    exact_matches = [item for item in items if item['name'] == filename]
                    if exact_matches:
                        return exact_matches[0]['path']
                    # Otherwise return the first match
                    return items[0]['path']
            elif r.status_code != 404:
                print(f"  ⚠️  Search failed with status {r.status_code}: {r.text}")
            
            # Small delay between strategies to avoid rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"  ❌ Strategy {i} failed: {e}")
            continue
    
    return None

def get_repository_tree(branch="main"):
    """Get the full repository tree to search locally"""
    try:
        url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        r = requests.get(url, headers=HEADERS)
        if r.status_code == 200:
            tree = r.json().get("tree", [])
            return {item['path'].split('/')[-1]: item['path'] for item in tree if item['type'] == 'blob'}
        else:
            print(f"⚠️  Could not fetch repository tree: HTTP {r.status_code}")
            return {}
    except Exception as e:
        print(f"⚠️  Error fetching repository tree: {e}")
        return {}

def download_file_from_repo(file_path, filename):
    """Enhanced download with better error handling"""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}?ref={args.branch}"
    
    try:
        r = requests.get(url, headers=HEADERS)
        if r.status_code == 200:
            file_data = r.json()
            
            # Handle large files
            if file_data.get('size', 0) > 1024 * 1024:  # 1MB
                print(f"⚠️  Large file detected: {file_path}")
                if 'download_url' in file_data:
                    # Download directly
                    download_r = requests.get(file_data['download_url'])
                    if download_r.status_code == 200:
                        content = download_r.text
                    else:
                        print(f"❌ Failed to download large file: {file_path}")
                        return False
                else:
                    print(f"❌ Large file without download URL: {file_path}")
                    return False
            else:
                # Decode base64 content
                content = base64.b64decode(file_data['content']).decode('utf-8')
            
            # Save to output directory with path preservation
            output_path = Path(args.output_dir) / filename
            # If there are path conflicts, use the full path
            if output_path.exists():
                safe_path = file_path.replace('/', '_')
                output_path = Path(args.output_dir) / safe_path
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Downloaded: {file_path} -> {output_path}")
            return True
            
        else:
            print(f"❌ Failed to download {file_path}: HTTP {r.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading {file_path}: {e}")
        return False

# --- Step 3: Enhanced search and download ---
print("🌳 Fetching repository tree for better matching...")
repo_tree = get_repository_tree(args.branch)

successful_downloads = 0
failed_searches = []

for filename in file_names:
    print(f"\n🔍 Searching for: {filename}")
    
    # First try: exact match in repository tree
    if filename in repo_tree:
        print(f"  ✅ Found in repository tree: {repo_tree[filename]}")
        if download_file_from_repo(repo_tree[filename], filename):
            successful_downloads += 1
        continue
    
    # Second try: fuzzy matching in repository tree
    fuzzy_matches = []
    for tree_filename, tree_path in repo_tree.items():
        if filename.lower() in tree_filename.lower() or tree_filename.lower() in filename.lower():
            fuzzy_matches.append((tree_filename, tree_path))
    
    if fuzzy_matches:
        print(f"  🎯 Found fuzzy matches in tree: {[match[0] for match in fuzzy_matches]}")
        # Use the best match (shortest name difference)
        best_match = min(fuzzy_matches, key=lambda x: abs(len(x[0]) - len(filename)))
        if download_file_from_repo(best_match[1], filename):
            successful_downloads += 1
        continue
    
    # Third try: GitHub search API
    path = search_file_multiple_strategies(filename)
    if path:
        if download_file_from_repo(path, filename):
            successful_downloads += 1
    else:
        print(f"  ❌ File not found: {filename}")
        failed_searches.append(filename)

# --- Summary ---
print(f"\n📊 Summary:")
print(f"   ✅ Successfully downloaded: {successful_downloads}/{len(file_names)} files")
if failed_searches:
    print(f"   ❌ Failed to find: {failed_searches}")
print(f"   📁 Files saved to: {Path(args.output_dir).absolute()}")