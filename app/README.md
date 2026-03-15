# RitS Transition State Generation App

A Streamlit web application for generating and visualizing molecular transition states using the RitS model, with optional IRC post-processing.

## Usage

```bash
conda activate rits

# From the repository root
streamlit run app/app.py
```

## Input

- **Reaction SMARTS**: Atom-mapped `R>>P` string (with explicit hydrogens)
- **Example reactions**: Built-in examples (amide hydrolysis, Diels-Alder, click chemistry, epoxidation, ester exchange, carbamate formation, E2 elimination)
- **Sampling parameters**: Number of steps (5-100), number of samples (1-50), molecular charge, stereo toggle

## Output

- **2D reaction scheme** (heavy atoms, no stereo annotations)
- **3D transition-state viewer** with sample selector (bonds shown = common bonds in both R and P)
- **Downloadable XYZ files** (single sample or all samples)
- **IRC trajectory viewer** with frame slider (reactant - TS - product)

## Requirements

### Model Files
- RitS checkpoint (`data/rits.ckpt`)
- RitS config (`scripts/conf/rits.yaml`)

### Python Packages

```bash
pip install -r app/requirements.txt
```

### IRC Post-Processing

IRC requires `pysisyphus` (included in `requirements.txt`) and `xtb`:

```bash
conda install -c conda-forge xtb
```
