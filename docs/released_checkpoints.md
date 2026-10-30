# Running the released RitS checkpoints

Install RitS using the environment instructions in the repository README, then
download [rits.ckpt](https://drive.google.com/file/d/1tmCmIuzgDsZYg3DeMeA0fcKDqvbCyqye/view),
[rits_finetuned.ckpt](https://drive.google.com/file/d/1fRNPWayFOqvee5Zvg89fEnwZNa7y90_c/view)
and [rits_tsdiff.ckpt](https://drive.google.com/file/d/1eetV9VqIFAe-dCOaITePVLE5B0yiFE5m/view)
from the [release data folder](https://drive.google.com/drive/folders/1J_b7U7KoP8nuAciDO76UpxVd3tX6VHYU)
into `data/`. Run the commands below from the repository root. The training HDF5
file is not needed for inference.

| Checkpoint | Training and intended use | Inference configuration | Input preprocessing |
|---|---|---|---|
| `rits.ckpt` | Original GFN2-xTB model; used for the broader reaction and selectivity examples | `scripts/conf/rits.yaml` | `--kekulize --add_stereo` |
| `rits_finetuned.ckpt` | GFN2-xTB pretraining followed by fine-tuning on the TSDiff/Grambow benchmark; reported as **RitS** in the benchmark comparison | `scripts/conf/rits.yaml` | `--kekulize --add_stereo` |
| `rits_tsdiff.ckpt` | Trained only on the TSDiff/Grambow benchmark; reported as **RitS-TSDiff** | `scripts/conf/ts1x_rits.yaml` | Neither flag; matches its benchmark preprocessing |

TSDiff/Grambow and Transition1x are distinct datasets. The `ts1x` configuration
and directory names are retained for compatibility. The older
`data/ts1x_rits.ckpt` is not the selected `rits_tsdiff.ckpt` release checkpoint.

`rits.yaml` also works for the fine-tuned checkpoint because it retains the
original model's 63-feature input representation. The benchmark-only checkpoint
uses 61 input features and needs `ts1x_rits.yaml`. These commands load the
checkpoint strictly and use EMA weights in evaluation mode.

## Generate transition-state guesses

The example uses atom-mapped reactant and product SMILES with explicit hydrogen
atoms. Maps must identify the same atoms on both sides; the two strings may use
different atom orders. No input 3D geometries are required.

```bash
reactant='[C:1]([c:2]1[n:3][o:4][n:5][n:6]1)([H:7])([H:8])[H:9]'
product='[C:1]1([H:7])([H:8])/[C:2](=[N:3]\[H:9])[N:6]1[N:5]=[O:4]'

# Original GFN2-xTB model
python scripts/sample_transition_state.py \
    --reactant_smi "$reactant" --product_smi "$product" \
    --config scripts/conf/rits.yaml --ckpt data/rits.ckpt \
    --kekulize --add_stereo --num_steps 25 \
    --n_samples 1 --batch_size 1 --device cpu --output rits_ts.xyz

# Pretrained model after benchmark fine-tuning
python scripts/sample_transition_state.py \
    --reactant_smi "$reactant" --product_smi "$product" \
    --config scripts/conf/rits.yaml --ckpt data/rits_finetuned.ckpt \
    --kekulize --add_stereo --num_steps 25 \
    --n_samples 1 --batch_size 1 --device cpu --output rits_finetuned_ts.xyz

# Model trained on the benchmark only
python scripts/sample_transition_state.py \
    --reactant_smi "$reactant" --product_smi "$product" \
    --config scripts/conf/ts1x_rits.yaml --ckpt data/rits_tsdiff.ckpt \
    --num_steps 25 --n_samples 1 --batch_size 1 \
    --device cpu --output rits_tsdiff_ts.xyz
```

Use `--device cuda` for an available GPU, or `--device auto` to select it
when available. Increase `--n_samples` to generate multiple guesses; all samples
are written to the specified XYZ file. Sampling is stochastic. For a charged
system, supply its net charge with `--charge`; the benchmark-only model was
trained on neutral systems. The coordinates are in angstrom and follow atom-map
order. These are generated guesses; stationary-point and IRC validation require
subsequent calculations.

## Checkpoint identity

Verify downloaded files with `sha256sum data/rits*.ckpt`:

```text
2c6fdacc13a304cb8bb4030881ff4198de647d8f8f7fdaa878414e2514be0668  rits.ckpt
fe6fec9d9d39e52b7046b8418dccc3fcd839ee904f1a8071102ec608374c911e  rits_finetuned.ckpt
1232e6017aa18be857aaf0c59c2089d42431d6e77f24819be3d7e52fc2d213a0  rits_tsdiff.ckpt
```

The selected fine-tuned and benchmark-only checkpoints are epochs 48 and 84,
respectively. The original checkpoint is the same model used by the existing
`data/rits.ckpt` examples.
