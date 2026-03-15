PYTHONPATH="./src" python scripts/sample_transition_state.py \
  --reaction_smarts "[C:1]([C:2]([O:3][C:4](=[O:5])[C:6]([C:7]([C:8]([H:17])([H:18])[H:19])=[O:9])([H:15])[H:16])([H:13])[H:14])([H:10])([H:11])[H:12].[O:20]([C:23]([H:21])([H:22])[H:24])[H:25]>>[C:1]([C:2]([O:3][C:4](=[O:5])[C:6]([H:15])([H:16])[H:25])([H:13])[H:14])([H:10])([H:11])[H:12].[C:7]([C:8]([H:17])([H:18])[H:19])(=[O:9])[O:20][C:23]([H:21])([H:22])[H:24]" \
  --config scripts/conf/rits.yaml \
  --ckpt data/rits.ckpt \
  --output examples/esterfication/sampled.xyz \
  --kekulize \
  --add_stereo \
  --num_steps 25 \
  --n_samples 10
