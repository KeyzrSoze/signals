import os
import sys


def create_structure():
    """
    Sets up a Production-Grade Directory Structure for Pharmacy AI.
    """
    # 1. Define the folder tree
    folders = [
        "config",
        "data/raw",              # Incoming dumps (NADAC, FDA JSON)
        "data/processed",        # Parquet files (Cleaned)
        "data/outputs",          # Final reports/graphs for clients
        "notebooks",             # Jupyter notebooks for testing
        "src",                   # Source Code
        "src/ingestion",         # Scripts to fetch data
        "src/processing",        # Cleaning logic
        "src/features",          # The Signal Generation Engine
        "src/models",            # AI/ML Training scripts
        "src/reporting",         # Code to generate PDFs/Decks
        "tests"                  # Unit tests
    ]

    # 2. Create folders
    print(f"🚀 Initializing Project in: {os.getcwd()}")
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        init_file = os.path.join(folder, "__init__.py")
        # Create __init__.py to make these Python packages
        if "data" not in folder and "notebooks" not in folder:
            with open(init_file, "w") as f:
                pass
        print(f"   Created: {folder}/")

    # 3. Create .gitignore (Crucial for Enterprise Security)
    gitignore_content = """
# Ignore Data (Too large, potential privacy issues)
data/
*.csv
*.parquet
*.zip

# Ignore Environment
venv/
__pycache__/
.DS_Store
.env

# Ignore IDE settings
.vscode/
"""
    with open(".gitignore", "w") as f:
        f.write(gitignore_content.strip())
    print("   Created: .gitignore")

    # 4. Create requirements.txt
    requirements = """
polars>=0.20.0
pandas>=2.0.0
requests>=2.31.0
scikit-learn>=1.3.0
xgboost>=2.0.0
pyarrow>=14.0.0
plotly>=5.18.0
jupyter>=1.0.0
"""
    with open("requirements.txt", "w") as f:
        f.write(requirements.strip())
    print("   Created: requirements.txt")

    print("\n✅ Setup Complete! You are ready for Phase 2.")


if __name__ == "__main__":
    create_structure()
