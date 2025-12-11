# BabyPlots Integration

This directory contains the BabyPlots JavaScript library for use in Jupyter notebooks.

## Setup

1. Download `babyplots.min.js` from the official BabyPlots website:
   - Visit: https://bp.bleb.li/
   - Or check: https://github.com/babyplots/babyplots/releases
   - Download the latest `babyplots.min.js` file

2. Replace the placeholder `babyplots.min.js` file in this directory with the downloaded file.

## Usage

In your Jupyter notebook or Google Colab:

```python
from src.protoInterpretation.visualization.babyplots import load_babyplots

# Load BabyPlots JS into the notebook
load_babyplots()

# Now you can use BabyPlots in your notebook
from babyplots import Babyplot
# ... your code ...
```

## Benefits

- No pip installation required
- No remote downloads (works offline)
- Faster loading (local file)
- Version control friendly

