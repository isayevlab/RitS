PYTHONPATH="./src" python scripts/sample_transition_state.py \
  --reaction_smarts "[C:11]([H:12])([H:13])([H:14])[N:15]=[C:16]=[O:17].[C:1](=[C:2]([C:3]([O:4][H:5])([H:9])[H:10])[H:8])([H:6])[H:7]>>[C:1](=[C:2]([C:3]([O:4][C:16]([N:15]([H:5])[C:11]([H:12])([H:13])[H:14])=[O:17])([H:9])[H:10])[H:8])([H:6])[H:7]" \
  --config scripts/conf/rits.yaml \
  --ckpt data/rits.ckpt \
  --output examples/carbomate/sampled.xyz \
  --kekulize \
  --add_stereo \
  --num_steps 25 \
  --n_samples 10
