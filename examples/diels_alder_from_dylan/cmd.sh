PYTHONPATH="./src" python scripts/sample_transition_state.py \
  --reaction_smarts "[C:12](#[C:13][C@@:14]1([H:29])[C:15]([H:30])([H:31])[N:16]([H:32])[C:17]([H:33])([H:34])[C:18]1([H:35])[H:36])[H:28].[C:1](=[C:2](/[C:3](=[C:4](\[C:5]([C:6](=[O:7])[O:8][C:9](=[O:10])[N:11]([H:26])[H:27])([H:24])[H:25])[H:23])[H:22])[H:21])([H:19])[H:20]>>[C:1]1([H:19])([H:20])[C:2]([H:21])=[C:3]([H:22])[C@:4]([C:5]([C:6](=[O:7])[O:8][C:9](=[O:10])[N:11]([H:26])[H:27])([H:24])[H:25])([H:23])[C:12]([H:28])=[C:13]1[C@@:14]1([H:29])[C:15]([H:30])([H:31])[N:16]([H:32])[C:17]([H:33])([H:34])[C:18]1([H:35])[H:36]" \
  --config scripts/conf/rits.yaml \
  --ckpt data/rits.ckpt \
  --output examples/diels_alder_from_dylan/sampled.xyz \
  --kekulize \
  --add_stereo \
  --num_steps 25 \
  --n_samples 10
