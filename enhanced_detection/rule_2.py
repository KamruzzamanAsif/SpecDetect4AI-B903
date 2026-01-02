import ast
import os
import networkx as nx

# ==========================================
# 1. THE "SMART" IMPORT RESOLVER
# ==========================================
def resolve_import_path(repo_root, current_file_dir, import_name, level=0):
    """
    Tries to find the actual .py file for an import string.
    Handles:
    - Absolute imports: 'import utils' -> checks repo_root/utils.py AND current_dir/utils.py
    - Relative imports: 'from . import utils' (level=1)
    - Submodules: 'import mypkg.core' -> mypkg/core.py
    """
    candidates = []
    
    # Convert 'mypkg.core' to 'mypkg/core'
    path_parts = import_name.split('.')
    relative_path = os.path.join(*path_parts)
    
    # STRATEGY 1: Check Relative to Current File (Script Mode / Relative Import)
    # If level > 0, we go up 'level' directories. 
    # If level == 0, we still check current dir because bad code often does 'import sibling'
    search_dir = current_file_dir
    for _ in range(level - 1): # Adjust for relative levels (.. import)
        search_dir = os.path.dirname(search_dir)
        
    # Check for file.py
    candidate_1 = os.path.join(search_dir, relative_path + ".py")
    # Check for package/__init__.py
    candidate_2 = os.path.join(search_dir, relative_path, "__init__.py")
    
    candidates.extend([candidate_1, candidate_2])

    # STRATEGY 2: Check Relative to Repo Root (Project Absolute Import)
    # e.g. 'import src.models' inside 'tests/test_a.py'
    candidate_3 = os.path.join(repo_root, relative_path + ".py")
    candidate_4 = os.path.join(repo_root, relative_path, "__init__.py")
    
    candidates.extend([candidate_3, candidate_4])
    
    # Return the first one that exists
    for path in candidates:
        if os.path.exists(path):
            # Normalize path to match node IDs
            return os.path.abspath(path)
            
    return None

# ==========================================
# 2. FILE VISITOR (Extracts Raw Imports)
# ==========================================
class ImportVisitor(ast.NodeVisitor):
    def __init__(self):
        self.imports = []       # List of (name, level)
        self.sets_seed = False
        self.uses_random = False

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append((alias.name, 0)) # Standard import (level 0)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module = node.module if node.module else ""
        self.imports.append((module, node.level)) # relative import (level > 0)
        self.generic_visit(node)
        
    def visit_Call(self, node):
        # Smell Detection Logic
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ['seed', 'manual_seed', 'set_seed']:
                self.sets_seed = True
            if node.func.attr in ['shuffle', 'rand', 'randn', 'sample', 'choice', 'randint']:
                self.uses_random = True
        self.generic_visit(node)

# ==========================================
# 3. BUILD GRAPH (The Robust Way)
# ==========================================
def build_resilient_graph(repo_path):
    G = nx.DiGraph()
    repo_path = os.path.abspath(repo_path)
    
    print(f"Scanning Repo: {repo_path}")
    
    # PASS 1: NODES (Scan all files)
    file_map = {} # Map abspath -> visitor data
    
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                abs_path = os.path.abspath(full_path)
                
                try:
                    with open(full_path, "r", encoding="utf-8", errors='ignore') as f:
                        tree = ast.parse(f.read())
                        visitor = ImportVisitor()
                        visitor.visit(tree)
                        
                        # Add Node (Use RelPath as ID for readability)
                        rel_id = os.path.relpath(abs_path, repo_path)
                        G.add_node(rel_id, 
                                   sets_seed=visitor.sets_seed, 
                                   uses_random=visitor.uses_random)
                        
                        file_map[rel_id] = {
                            "abs_path": abs_path,
                            "imports": visitor.imports,
                            "dir": os.path.dirname(abs_path)
                        }
                except Exception as e:
                    # Skip files with syntax errors (common in scraped repos)
                    pass

    # PASS 2: EDGES (Resolve Imports)
    print(f"Resolving dependencies for {len(G.nodes)} files...")
    
    for node_id in G.nodes:
        data = file_map.get(node_id)
        if not data: continue
        
        for imp_name, level in data['imports']:
            # Skip empty imports
            if not imp_name and level == 0: continue
            
            # Try to find the file
            target_path = resolve_import_path(repo_path, data['dir'], imp_name, level)
            
            if target_path:
                target_rel_id = os.path.relpath(target_path, repo_path)
                
                # Check if this file is actually in our graph
                if target_rel_id in G.nodes:
                    G.add_edge(node_id, target_rel_id) # node imports target

    return G

# ==========================================
# 4. ANALYSIS (Reachability)
# ==========================================
def analyze_fragility(G):
    # Find Seeders
    seeders = [n for n, attr in G.nodes(data=True) if attr.get('sets_seed')]
    print(f"Seed Sources ({len(seeders)}): {seeders}")
    
    fragile_files = []
    
    for node, attr in G.nodes(data=True):
        if not attr.get('uses_random'):
            continue
            
        # Is Safe?
        is_safe = False
        
        # 1. Self Seeded
        if attr.get('sets_seed'): 
            is_safe = True
            
        # 2. Inherits Seed (Imports a seeder)
        # Check Descendants (dependencies)
        if not is_safe:
            dependencies = nx.descendants(G, node)
            if not set(dependencies).isdisjoint(seeders):
                is_safe = True
                
        # 3. Context Seed (Imported by a seeder)
        # Check Ancestors (callers)
        if not is_safe:
            callers = nx.ancestors(G, node)
            if not set(callers).isdisjoint(seeders):
                is_safe = True
                
        if not is_safe:
            fragile_files.append(node)
            
    return fragile_files

# ==========================================
# 5. TEST RUNNER
# ==========================================
# Change this to your repo path!
# repo_path = "/path/to/my_repo"

# For Demo: Let's create the 'Problematic' repo structure again
def create_messy_repo():
    base = "messy_repo"
    os.makedirs(base, exist_ok=True)
    
    # 1. Script that imports sibling WITHOUT package notation (The Grimp Killer)
    with open(f"{base}/main.py", "w") as f:
        f.write("import utils\nimport numpy as np\nnp.random.seed(42)\nutils.run()")
        
    # 2. The Sibling
    with open(f"{base}/utils.py", "w") as f:
        f.write("import numpy as np\ndef run():\n    print(np.random.rand())")
        
    # 3. An Isolated Script
    with open(f"{base}/orphan.py", "w") as f:
        f.write("import numpy as np\nprint(np.random.rand())")


# create_messy_repo()
G = build_resilient_graph("messy_repo")
fragile = analyze_fragility(G)

print("-" * 30)
print(f"Fragile Files: {fragile}")
print("-" * 30)
# Expected: orphan.py ONLY. (utils.py is safe because main.py imports it!)