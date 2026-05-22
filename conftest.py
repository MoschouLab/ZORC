import sys
from pathlib import Path

# Ensure project root is on sys.path so `import api.main` works regardless
# of where pytest is invoked from (local, CI, tox, etc.).
sys.path.insert(0, str(Path(__file__).parent))
