"""Make the asr_bench root importable so `import speech_base`, `import adapters`, etc.
resolve whether tests run from here or from a parent directory.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
