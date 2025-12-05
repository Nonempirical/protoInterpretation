# Interactive Animated UMAP Colab Example

This shows how to create an interactive UI for selecting runs and controlling animation speed.

## Colab Cell Code

```python
import os
import ipywidgets as widgets
from IPython.display import display
from google.colab import drive

from protoInterpretation import (
    find_runs_directory,
    scan_runs,
    load_embeddings_from_runs,
    compute_global_umap,
    plot_animated_umap,
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
# 4. Build interactive UI
# ============================================================

# Multi-select dropdown for runs
run_selector = widgets.SelectMultiple(
    options=run_names,
    value=run_names[:2] if len(run_names) >= 2 else run_names[:1],
    description="Select runs:",
    disabled=False,
    layout=widgets.Layout(width="100%", height="200px"),
)

# Slider for animation speed (frame duration in milliseconds)
# Lower values = faster animation
speed_slider = widgets.IntSlider(
    value=100,
    min=50,
    max=500,
    step=50,
    description="Speed (ms/frame):",
    tooltip="Lower values = faster animation",
    continuous_update=False,
    layout=widgets.Layout(width="60%"),
)

# Button to generate animation
animate_button = widgets.Button(
    description="Generate Animation",
    button_style="info",
    layout=widgets.Layout(width="200px"),
)

# Output area for the plot
out = widgets.Output(layout=widgets.Layout(height="700px"))

# ============================================================
# 5. Animation callback
# ============================================================
def on_animate_clicked(_):
    with out:
        out.clear_output()
        
        selected_runs = list(run_selector.value)
        if not selected_runs:
            print("Please select at least one run.")
            return
        
        print(f"Loading {len(selected_runs)} runs: {selected_runs}")
        
        # Load embeddings
        emb_per_run, min_T, D = load_embeddings_from_runs(selected_runs, runs)
        
        total_points = sum(emb.shape[0] * min_T for emb in emb_per_run.values())
        print(f"Combined N*T points: {total_points}, min_T={min_T}, D={D}")
        
        # Compute global UMAP
        print("\nRunning PCA + UMAP on combined embeddings...")
        embeddings_2d, run_labels, step_labels = compute_global_umap(
            emb_per_run,
            max_steps=min_T,
        )
        
        # Get animation speed from slider (convert to frame duration)
        frame_duration = speed_slider.value
        
        # Create and display animated plot
        print(f"\nGenerating animation (speed: {frame_duration}ms per frame)...")
        fig = plot_animated_umap(
            embeddings_2d=embeddings_2d,
            run_labels=run_labels,
            step_labels=step_labels,
            title=f"Global UMAP horizon movie ({', '.join(selected_runs)})",
            animation_frame_duration=frame_duration,
        )
        
        fig.show()
        print("\n✅ Animation complete!")

animate_button.on_click(on_animate_clicked)

# ============================================================
# 6. Display UI
# ============================================================
ui_box = widgets.VBox([
    widgets.HTML("<h3>Animated UMAP Visualization</h3>"),
    widgets.HBox([
        widgets.VBox([
            run_selector,
            speed_slider,
            animate_button,
        ], layout=widgets.Layout(width="40%")),
        out,
    ]),
])

display(ui_box)
```

## Features

- **Multi-select dropdown**: Select one or more runs to animate together
- **Speed slider**: Control animation playback speed (50-500ms per frame)
- **Interactive**: Click "Generate Animation" to create the plot with your selected settings
- **Real-time feedback**: Shows progress messages while loading and computing

## Usage Tips

- Select multiple runs to compare them side-by-side in the same animation
- Lower speed values (50-100ms) = faster animation
- Higher speed values (300-500ms) = slower, more detailed viewing
- The animation will show all selected runs with different colors

