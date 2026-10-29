import os
import sys
from pathlib import Path

# Before anything imports huggingface_hub: the tests build their checkpoints locally and never use the Hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
