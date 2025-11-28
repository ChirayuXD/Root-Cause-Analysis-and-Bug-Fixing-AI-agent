import os
import argparse
import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import hashlib
import re

# Vector/embedding libraries
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Code parsing libraries
try:
    import javalang
    JAVA_PARSER_AVAILABLE = True
except ImportError:
    print("⚠️  javalang not available. Install with: pip install javalang")
    JAVA_PARSER_AVAILABLE = False

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Index code files into FAISS for semantic search")
parser.add_argument("--input_dir", default="downloaded_files", help="Directory containing code files")
parser.add_argument("--output_dir", default="faiss_index", help="Directory to save FAISS index and metadata")
parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence transformer model")
parser.add_argument("--chunk_size", type=int, default=500, help="Characters per chunk")
parser.add_argument("--chunk_overlap", type=int, default=50, help="Overlap between chunks")
parser.add_argument("--index_type", choices=['flat', 'ivf', 'hnsw'], default='flat', help="FAISS index type")
parser.add_argument("--rebuild", action='store_true', help="Force rebuild of existing index")

args = parser.parse_args()

# Create output directory
Path(args.output_dir).mkdir(exist_ok=True)

class CodeChunker:
    """Smart code chunking that respects code structure"""
    
    def __init__(self, chunk_size=500, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_java_file(self, content: str, filename: str) -> List[Dict]:
        """Chunk Java file respecting method/class boundaries"""
        chunks = []
        
        if JAVA_PARSER_AVAILABLE:
            try:
                # Try to parse Java AST for smart chunking
                tree = javalang.parse.parse(content)
                chunks.extend(self._extract_java_methods(tree, content, filename))
                chunks.extend(self._extract_java_classes(tree, content, filename))
            except Exception as e:
                print(f"  ⚠️  Java parsing failed for {filename}, using text chunking: {e}")
                chunks = self._chunk_by_text(content, filename)
        else:
            chunks = self._chunk_by_text(content, filename)
        
        return chunks
    
    def _extract_java_methods(self, tree, content: str, filename: str) -> List[Dict]:
        """Extract individual methods as chunks"""
        chunks = []
        lines = content.split('\n')
        
        for path, node in tree:
            if isinstance(node, javalang.tree.MethodDeclaration):
                try:
                    start_line = node.position.line - 1 if node.position else 0
                    # Find method end (simple heuristic)
                    brace_count = 0
                    end_line = start_line
                    
                    for i in range(start_line, len(lines)):
                        line = lines[i]
                        brace_count += line.count('{') - line.count('}')
                        if brace_count == 0 and '{' in line:
                            end_line = i
                            break
                    
                    method_content = '\n'.join(lines[start_line:end_line + 1])
                    
                    if len(method_content.strip()) > 10:  # Skip tiny methods
                        chunks.append({
                            'content': method_content,
                            'type': 'method',
                            'name': node.name,
                            'filename': filename,
                            'start_line': start_line + 1,
                            'end_line': end_line + 1,
                            'id': f"{filename}:method:{node.name}:{start_line}"
                        })
                except Exception as e:
                    print(f"    ⚠️  Error extracting method {node.name}: {e}")
        
        return chunks
    
    def _extract_java_classes(self, tree, content: str, filename: str) -> List[Dict]:
        """Extract class-level information"""
        chunks = []
        
        for path, node in tree:
            if isinstance(node, javalang.tree.ClassDeclaration):
                try:
                    # Get class signature and fields
                    class_info = f"Class: {node.name}\n"
                    if node.extends:
                        class_info += f"Extends: {node.extends.name}\n"
                    if node.implements:
                        implements = [impl.name for impl in node.implements]
                        class_info += f"Implements: {', '.join(implements)}\n"
                    
                    # Add field declarations
                    for field in node.fields:
                        for declarator in field.declarators:
                            class_info += f"Field: {field.type.name} {declarator.name}\n"
                    
                    chunks.append({
                        'content': class_info,
                        'type': 'class_info',
                        'name': node.name,
                        'filename': filename,
                        'start_line': node.position.line if node.position else 1,
                        'end_line': node.position.line if node.position else 1,
                        'id': f"{filename}:class:{node.name}"
                    })
                except Exception as e:
                    print(f"    ⚠️  Error extracting class {node.name}: {e}")
        
        return chunks
    
    def _chunk_by_text(self, content: str, filename: str) -> List[Dict]:
        """Fallback text-based chunking with code awareness"""
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_size = 0
        chunk_id = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            line_size = len(line) + 1  # +1 for newline
            
            # If adding this line would exceed chunk size, save current chunk
            if current_size + line_size > self.chunk_size and current_chunk:
                chunk_content = '\n'.join(current_chunk)
                chunks.append({
                    'content': chunk_content,
                    'type': 'text_chunk',
                    'name': f"chunk_{chunk_id}",
                    'filename': filename,
                    'start_line': i - len(current_chunk) + 1,
                    'end_line': i,
                    'id': f"{filename}:chunk:{chunk_id}"
                })
                
                # Start new chunk with overlap
                overlap_lines = max(1, self.overlap // 50)  # Rough estimate
                current_chunk = current_chunk[-overlap_lines:] if overlap_lines < len(current_chunk) else []
                current_size = sum(len(line) + 1 for line in current_chunk)
                chunk_id += 1
            
            current_chunk.append(line)
            current_size += line_size
            i += 1
        
        # Add final chunk
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append({
                'content': chunk_content,
                'type': 'text_chunk',
                'name': f"chunk_{chunk_id}",
                'filename': filename,
                'start_line': len(lines) - len(current_chunk) + 1,
                'end_line': len(lines),
                'id': f"{filename}:chunk:{chunk_id}"
            })
        
        return chunks

class FAISSIndexer:
    """FAISS indexer for code chunks"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"🤖 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"   📐 Embedding dimension: {self.dimension}")
        
    def create_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """Create embeddings for all chunks"""
        print(f"🔄 Creating embeddings for {len(chunks)} chunks...")
        
        # Prepare texts for embedding
        texts = []
        for chunk in chunks:
            # Create rich text representation
            text = f"File: {chunk['filename']}\n"
            text += f"Type: {chunk['type']}\n"
            if chunk['type'] in ['method', 'class_info']:
                text += f"Name: {chunk['name']}\n"
            text += f"Content:\n{chunk['content']}"
            texts.append(text)
        
        # Create embeddings in batches
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            embeddings.append(batch_embeddings)
            print(f"   📊 Processed {min(i + batch_size, len(texts))}/{len(texts)} chunks")
        
        return np.vstack(embeddings).astype('float32')
    
    def create_index(self, embeddings: np.ndarray, index_type: str = 'flat') -> faiss.Index:
        """Create FAISS index"""
        print(f"🗂️  Creating FAISS index (type: {index_type})...")
        
        if index_type == 'flat':
            index = faiss.IndexFlatIP(self.dimension)  # Inner Product (cosine similarity)
        elif index_type == 'ivf':
            quantizer = faiss.IndexFlatIP(self.dimension)
            nlist = min(100, max(1, embeddings.shape[0] // 10))  # Adaptive nlist
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            print(f"   🎯 Training IVF index with {nlist} clusters...")
            index.train(embeddings)
        elif index_type == 'hnsw':
            index = faiss.IndexHNSWFlat(self.dimension, 32)
            index.hnsw.efConstruction = 200
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        print(f"   📥 Adding {embeddings.shape[0]} vectors to index...")
        index.add(embeddings)
        
        print(f"   ✅ Index created with {index.ntotal} vectors")
        return index

def process_files(input_dir: str) -> List[Dict]:
    """Process all code files in directory"""
    chunker = CodeChunker(args.chunk_size, args.chunk_overlap)
    all_chunks = []
    
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Find all code files
    code_extensions = {'.java', '.py', '.js', '.ts', '.cpp', '.c', '.h', '.hpp', '.cs', '.go', '.rs'}
    code_files = []
    
    for ext in code_extensions:
        code_files.extend(input_path.glob(f"**/*{ext}"))
    
    if not code_files:
        print("⚠️  No code files found in input directory")
        return []
    
    print(f"📁 Found {len(code_files)} code files")
    
    for file_path in code_files:
        print(f"   📄 Processing: {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                print(f"     ⚠️  Empty file, skipping")
                continue
            
            # Choose chunking strategy based on file type
            if file_path.suffix == '.java':
                chunks = chunker.chunk_java_file(content, file_path.name)
            else:
                chunks = chunker._chunk_by_text(content, file_path.name)
            
            print(f"     ✅ Created {len(chunks)} chunks")
            all_chunks.extend(chunks)
            
        except Exception as e:
            print(f"     ❌ Error processing {file_path.name}: {e}")
    
    return all_chunks

def save_index_and_metadata(index: faiss.Index, chunks: List[Dict], output_dir: str, model_name: str):
    """Save FAISS index and metadata"""
    output_path = Path(output_dir)
    
    # Save FAISS index
    index_file = output_path / "code_index.faiss"
    faiss.write_index(index, str(index_file))
    print(f"💾 FAISS index saved to: {index_file}")
    
    # Save metadata
    metadata = {
        'chunks': chunks,
        'model_name': model_name,
        'total_chunks': len(chunks),
        'index_type': args.index_type,
        'chunk_size': args.chunk_size,
        'chunk_overlap': args.chunk_overlap
    }
    
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"📋 Metadata saved to: {metadata_file}")
    
    # Save search helper
    search_script = output_path / "search_index.py"
    with open(search_script, 'w', encoding='utf-8') as f:
        f.write(create_search_script())
    print(f"🔍 Search script saved to: {search_script}")

def create_search_script() -> str:
    """Generate a search script for the created index"""
    return '''#!/usr/bin/env python3
"""
Search script for FAISS code index
Usage: python search_index.py "your search query"
"""
import sys
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def load_index_and_metadata():
    """Load FAISS index and metadata"""
    index = faiss.read_index("code_index.faiss")
    with open("metadata.json", 'r') as f:
        metadata = json.load(f)
    model = SentenceTransformer(metadata['model_name'])
    return index, metadata, model

def search(query: str, top_k: int = 5):
    """Search for similar code chunks"""
    index, metadata, model = load_index_and_metadata()
    
    # Create query embedding
    query_embedding = model.encode([query]).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    # Search
    scores, indices = index.search(query_embedding, top_k)
    
    print(f"🔍 Search results for: '{query}'\\n")
    
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx == -1:  # No more results
            break
            
        chunk = metadata['chunks'][idx]
        print(f"📍 Result {i+1} (Score: {score:.4f})")
        print(f"   📁 File: {chunk['filename']}")
        print(f"   📝 Type: {chunk['type']}")
        if chunk['type'] in ['method', 'class_info']:
            print(f"   🏷️  Name: {chunk['name']}")
        print(f"   📏 Lines: {chunk['start_line']}-{chunk['end_line']}")
        print(f"   📄 Content preview:")
        content_preview = chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
        print(f"      {content_preview}")
        print()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python search_index.py \\"your search query\\"")
        sys.exit(1)
    
    query = sys.argv[1]
    search(query)
'''

def main():
    """Main execution function"""
    print("🚀 Starting FAISS Code Indexer")
    print(f"   📂 Input directory: {args.input_dir}")
    print(f"   📊 Output directory: {args.output_dir}")
    print(f"   🤖 Model: {args.model}")
    print(f"   📏 Chunk size: {args.chunk_size}")
    print(f"   🔗 Index type: {args.index_type}")
    
    # Check if index already exists
    index_file = Path(args.output_dir) / "code_index.faiss"
    if index_file.exists() and not args.rebuild:
        print(f"⚠️  Index already exists at {index_file}")
        print("   Use --rebuild to force recreation")
        return
    
    # Step 1: Process files and create chunks
    print("\\n📄 Step 1: Processing files...")
    chunks = process_files(args.input_dir)
    
    if not chunks:
        print("❌ No chunks created. Exiting.")
        return
    
    print(f"✅ Created {len(chunks)} total chunks")
    
    # Step 2: Create embeddings
    print("\\n🔄 Step 2: Creating embeddings...")
    indexer = FAISSIndexer(args.model)
    embeddings = indexer.create_embeddings(chunks)
    
    # Step 3: Build FAISS index
    print("\\n🗂️  Step 3: Building FAISS index...")
    index = indexer.create_index(embeddings, args.index_type)
    
    # Step 4: Save everything
    print("\\n💾 Step 4: Saving index and metadata...")
    save_index_and_metadata(index, chunks, args.output_dir, args.model)
    
    print("\\n🎉 Indexing complete!")
    print(f"   📊 Total chunks indexed: {len(chunks)}")
    print(f"   🗂️  Index file: {Path(args.output_dir) / 'code_index.faiss'}")
    print(f"   🔍 To search: cd {args.output_dir} && python search_index.py \"your query\"")

if __name__ == "__main__":
    main()