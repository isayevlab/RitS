PYTHONPATH="./src" python scripts/sample_transition_state.py \
  --reaction_smarts "[C:1]([C:2](=[O:3])[O:4][C:5]([C:6]([C:7]([N:8]=[N+:9]=[N-:10])([H:18])[H:19])([H:16])[H:17])([H:14])[H:15])([H:11])([H:12])[H:13].[O:20]([C:21]([C:22]#[C:23][H:27])([H:25])[H:26])[H:24]>>[C@:1]([C:2](=[O:3])[O:4][C@@:5]([C@@:6]([C@@:7]([N:8]1:[N:9]:[N:10]:[C:22]([C@:21]([O:20][H:24])([H:25])[H:26]):[C:23]:1[H:27])([H:18])[H:19])([H:16])[H:17])([H:14])[H:15])([H:11])([H:12])[H:13]" \
  --config scripts/conf/rits.yaml \
  --ckpt data/rits.ckpt \
  --output examples/click/sampled.xyz \
  --kekulize \
  --add_stereo \
  --num_steps 25 \
  --n_samples 10
