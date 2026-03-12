# LoQI Conformer Generation App

A Streamlit web application for generating and analyzing molecular conformers using LoQI diffusion and flow-matching models.

## Usage

```bash
# Activate environment
conda activate loqi

# Run the app
streamlit run app.py
```

## Input

- **SMILES string**: Molecular structure (defaults to a "good" thalidomide example)
- **Number of conformers**: 1-20 conformers to generate
- **Model type**: Diffusion or Flow Matching
- **Sampling steps**:
  - Diffusion: up to 25
  - Flow Matching: up to 100
- **Postprocessing mode**:
  - `none`
  - `optimization`
  - `optimization + irmsd unique set selection`

## Output

- **3D visualization** of the best conformer (topology + stereochemistry preserved)
- **Energy statistics** (relative to minimum energy)
- **Preservation metrics** (topology %, stereochemistry %)
- **Generation time per structure** (seconds)
- **Conformer table** with detailed analysis
- **Downloadable SDF files** with energy annotations

## Requirements

### Model Files
- LoQI model checkpoint (`data/loqi.ckpt`)
- LoQI flow checkpoint (`data/loqi_flow.ckpt`)
- AIMNet2 model (`src/megalodon/metrics/aimnet2/cpcm_model/wb97m_cpcms_v2_0.jpt`)
- ChEMBL3D dataset (`data/chembl3d_stereo/`)

### Python Packages
Additional requirements for the app. 
```bash
pip install -r app/requirements.txt
```
