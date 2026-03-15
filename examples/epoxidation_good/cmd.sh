PYTHONPATH="./src" python scripts/sample_transition_state.py \
  --reaction_smarts "[H:11][O:19][C:20](=[C:22]([H:21])[H:23])[C:24]([H:25])([H:26])[H:27].[O:1]([C:2](=[O:3])[C:4]([c:5]1[c:6]([H:14])[c:7]([H:15])[c:8]([H:16])[c:9]([H:17])[c:10]1[H:18])([H:12])[H:13])[H:28]>>[O:19]1[C:20]([C:24]([H:25])([H:26])[H:27])([H:28])[C:22]1([H:21])[H:23].[O:1]=[C:2]([O:3][H:11])[C:4]([c:5]1[c:6]([H:14])[c:7]([H:15])[c:8]([H:16])[c:9]([H:17])[c:10]1[H:18])([H:12])[H:13]" \
  --config scripts/conf/rits.yaml \
  --ckpt data/rits.ckpt \
  --output examples/epoxidation_good/sampled.xyz \
  --kekulize \
  --add_stereo \
  --num_steps 25 \
  --n_samples 10
