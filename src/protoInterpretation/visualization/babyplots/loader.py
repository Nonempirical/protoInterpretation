"""Helper to load BabyPlots JS into Jupyter notebooks."""

import os
from IPython.display import HTML


def load_babyplots():
    """
    Loads BabyPlots JS into the current notebook.
    
    This function reads the bundled babyplots.min.js file and injects it
    into the notebook, making BabyPlots available globally without requiring
    remote downloads or pip installation.
    
    Returns:
        IPython.display.HTML: HTML script tag containing BabyPlots JS code
    """
    js_path = os.path.join(os.path.dirname(__file__), "babyplots.min.js")
    
    if not os.path.exists(js_path):
        raise FileNotFoundError(
            f"BabyPlots JS file not found at {js_path}. "
            "Please ensure babyplots.min.js is present in the babyplots directory."
        )
    
    with open(js_path, "r", encoding="utf-8") as f:
        js_code = f.read()
    
    return HTML(f"<script>{js_code}</script>")

