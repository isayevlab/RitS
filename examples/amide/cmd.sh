PYTHONPATH="./src" python scripts/sample_transition_state.py \
  --reaction_smarts "[C:1]([N:2]([C:4]([C:3]([H:10])([H:11])[H:12])=[O:13])[H:8])([H:5])([H:6])[H:7].[H:9][O:14][H:15]>>[C:1]([N:2]([H:8])[H:9])([H:5])([H:6])[H:7].[C:3]([C:4](=[O:13])[O:14][H:15])([H:10])([H:11])[H:12]" \
  --config scripts/conf/rits.yaml \
  --ckpt data/rits.ckpt \
  --output examples/amide/sampled.xyz \
  --kekulize \
  --add_stereo \
  --num_steps 25 \
  --n_samples 10
