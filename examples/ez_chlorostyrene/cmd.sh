# E pathway
PYTHONPATH="./src" python scripts/sample_transition_state.py \
  --reaction_smarts "[C:1]1([C:7]([C:8]([Cl:9])([Cl:10])[H:18])([H:16])[H:17])=[C:2]([H:11])[C:3]([H:12])=[C:4]([H:13])[C:5]([H:14])=[C:6]1[H:15]>>[C:1]1([C:7](=[C:8]([Cl:9])[H:18])[H:16])=[C:2]([H:11])[C:3]([H:12])=[C:4]([H:13])[C:5]([H:14])=[C:6]1[H:15].[Cl:10][H:17]" \
  --config scripts/conf/rits.yaml \
  --ckpt data/rits.ckpt \
  --output examples/ez_chlorostyrene/e_chlorostyrene.xyz \
  --kekulize --add_stereo --num_steps 25 --n_samples 100

# Z pathway
PYTHONPATH="./src" python scripts/sample_transition_state.py \
  --reaction_smarts "[C:1]1([C:7]([C:8]([Cl:9])([Cl:10])[H:18])([H:16])[H:17])=[C:2]([H:11])[C:3]([H:12])=[C:4]([H:13])[C:5]([H:14])=[C:6]1[H:15]>>[C:1]1([C:7](=[C:8]([Cl:9])[H:18])[H:16])=[C:2]([H:11])[C:3]([H:12])=[C:4]([H:13])[C:5]([H:14])=[C:6]1[H:15].[Cl:10][H:17]" \
  --config scripts/conf/rits.yaml \
  --ckpt data/rits.ckpt \
  --output examples/ez_chlorostyrene/z_chlorostyrene.xyz \
  --kekulize --add_stereo --num_steps 25 --n_samples 100
