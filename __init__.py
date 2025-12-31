"""Top-level package for comfyui_olafrv custom nodes"""

import os
import sys

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Add ComfyUI root to path (go up twice from current directory)
comfy_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, comfy_root)

from src.comfyui_olafrv.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

__author__ = """Olaf Reitmaier Veracierta"""
__email__ = "olafrv@gmail.com"
__version__ = "0.0.2"
__license__ = "MIT"
