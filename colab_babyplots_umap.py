"""
Google Colab cell for BabyPlots UMAP visualization.

This cell creates an interactive interface to visualize UMAP embeddings
from multiple protoInterpretation runs using BabyPlots.

Usage:
    Copy this entire cell into a Google Colab notebook and run it.
"""

# ============================================================
# Setup: Load BabyPlots
# ============================================================
from src.protoInterpretation.visualization.babyplots import load_babyplots
load_babyplots()

# ============================================================
# Imports
# ============================================================
import os
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, HTML
from google.colab import drive
from babyplots import Babyplot

from src.protoInterpretation import (
    find_runs_directory,
    scan_runs,
    load_embeddings_from_runs,
    compute_global_umap,
)

# ============================================================
# 1. Mount Drive
# ============================================================
drive.mount('/content/drive', force_remount=False)

# ============================================================
# 2. Locate protoInterpretation-runs folder
# ============================================================
PROJECT_FOLDER_NAME = "protoInterpretation-runs"
search_roots = [
    "/content/drive/MyDrive",
    "/content/drive/Shareddrives",
]

BASE_RUN_DIR = None
for root in search_roots:
    found = find_runs_directory(root, folder_name=PROJECT_FOLDER_NAME)
    if found:
        BASE_RUN_DIR = found
        break

if BASE_RUN_DIR is None:
    raise RuntimeError(
        f"Could not find a folder named '{PROJECT_FOLDER_NAME}' in MyDrive or SharedDrives."
    )

print("Using runs folder:", BASE_RUN_DIR)

# ============================================================
# 3. Scan for runs
# ============================================================
runs = scan_runs(BASE_RUN_DIR)
if not runs:
    raise RuntimeError(f"No runs with batch.npz found under {BASE_RUN_DIR}")

run_names = sorted(runs.keys())
print(f"Found {len(run_names)} runs:")
for r in run_names:
    print("  -", r)

# ============================================================
# 4. UI: 5 dropdown menus
# ============================================================
run_options = [""] + run_names

dropdowns = [
    widgets.Dropdown(
        options=run_options,
        value=(run_names[i] if i < len(run_names) else ""),
        description=f"Run {chr(65+i)}:",
        layout=widgets.Layout(width="45%")
    )
    for i in range(5)
]

generate_button = widgets.Button(
    description="Generate BabyPlots Animation",
    button_style="info",
    layout=widgets.Layout(width="260px"),
)

out = widgets.Output(layout=widgets.Layout(width="100%", height="750px"))

# ============================================================
# 5. BabyPlots callback
# ============================================================
def on_generate_clicked(_):
    with out:
        out.clear_output()

        # Collect selected runs
        selected_runs = []
        for d in dropdowns:
            if d.value:
                selected_runs.append(d.value)

        # Remove duplicates while preserving order
        selected_runs = list(dict.fromkeys(selected_runs))

        if len(selected_runs) < 2:
            print("⚠️ Please select at least 2 runs.")
            return

        print(f"Loading {len(selected_runs)} runs: {selected_runs}")

        emb_per_run, min_T, D = load_embeddings_from_runs(selected_runs, runs)
        total_points = sum(emb.shape[0] * min_T for emb in emb_per_run.values())
        print(f"Combined N*T points: {total_points}, min_T={min_T}, D={D}")

        print("\nRunning PCA + UMAP on combined embeddings...")
        embeddings_2d, run_labels, step_labels = compute_global_umap(
            emb_per_run, max_steps=min_T
        )

        print("UMAP complete. Plotting BabyPlots visualization...")

        # -----------------------------
        # Prepare DataFrame for BabyPlots
        # -----------------------------
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'z': np.zeros(len(embeddings_2d)),  # Flat plane for 2D UMAP
            'run': run_labels,
            'time': step_labels,  # Time column for animation
        })

        # -----------------------------
        # BabyPlots animation
        # -----------------------------
        bp = Babyplot()
        
        # Use add_plot_from_dataframe with time in options
        bp.add_plot_from_dataframe(
            dataframe=df,
            plot_type="pointCloud",
            coord_columns=["x", "y", "z"],
            color_by="categories",
            color_var="run",  # Color by run name
            options={
                "time": "time",  # Column name for animation timeline
                "colorScale": "Set1",  # Categorical color scale
                "pointSize": 3,
                "opacity": 0.85,
                "title": "BabyPlots UMAP Horizon Animation: " + ", ".join(selected_runs)
            }
        )

        display(bp)

        print("\n✅ BabyPlots animation ready!")

generate_button.on_click(on_generate_clicked)

# ============================================================
# 6. Display UI
# ============================================================
ui = widgets.VBox([
    widgets.HTML("<h3>BabyPlots — Global UMAP Horizon Animation</h3>"),
    widgets.HBox([dropdowns[0], dropdowns[1]]),
    widgets.HBox([dropdowns[2], dropdowns[3]]),
    widgets.HBox([dropdowns[4]]),
    generate_button,
    out,
])

display(ui)

