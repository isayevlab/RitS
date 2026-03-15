# Hajos-Parrish-like proline organocatalysis (5-step scaffold, v2)

# Step 1.1: step11_complex_1
PYTHONPATH="./src" python scripts/sample_transition_state.py \
  --reaction_smarts "[C:1]([C:2](=[O:3])[C:4]([C:5]([C:6]1([C:7]([H:21])([H:22])[H:23])[C:8](=[O:10])[C:9]([H:24])([H:25])[C:11]([H:26])([H:27])[C:12]1=[O:13])([H:19])[H:20])([H:17])[H:18])([H:14])([H:15])[H:16].[O:28]=[C:29]([O:30][H:36])[C@:31]1([H:37])[C:32]([H:38])([H:39])[C:33]([H:40])([H:41])[C:34]([H:42])([H:43])[N:35]1[H:44]>>[C:1]([C:2]([O:3][H:44])([C:4]([C:5]([C:6]1([C:7]([H:21])([H:22])[H:23])[C:8](=[O:10])[C:9]([H:24])([H:25])[C:11]([H:26])([H:27])[C:12]1=[O:13])([H:19])[H:20])([H:17])[H:18])[N:35]1[C@@:31]([C:29](=[O:28])[O:30][H:36])([H:37])[C:32]([H:38])([H:39])[C:33]([H:40])([H:41])[C:34]1([H:42])[H:43])([H:14])([H:15])[H:16]" \
  --config scripts/conf/rits.yaml \
  --ckpt data/rits.ckpt \
  --output examples/organocatalysis_hajos/step11_complex_1.xyz \
  --kekulize \
  --add_stereo \
  --num_steps 25 \
  --n_samples 100

# Step 1.2: step12_complex_2
PYTHONPATH="./src" python scripts/sample_transition_state.py \
  --reaction_smarts "[C:1]([C:2]([O:3][H:44])([C:4]([C:5]([C:6]1([C:7]([H:21])([H:22])[H:23])[C:8](=[O:10])[C:9]([H:24])([H:25])[C:11]([H:26])([H:27])[C:12]1=[O:13])([H:19])[H:20])([H:17])[H:18])[N:35]1[C@@:31]([C:29](=[O:28])[O:30][H:36])([H:37])[C:32]([H:38])([H:39])[C:33]([H:40])([H:41])[C:34]1([H:42])[H:43])([H:14])([H:15])[H:16]>>[C:1](=[C:2]([C:4]([C:5]([C:6]1([C:7]([H:21])([H:22])[H:23])[C:8](=[O:10])[C:9]([H:24])([H:25])[C:11]([H:26])([H:27])[C:12]1=[O:13])([H:19])[H:20])([H:17])[H:18])[N:35]1[C@@:31]([C:29](=[O:28])[O:30][H:36])([H:37])[C:32]([H:38])([H:39])[C:33]([H:40])([H:41])[C:34]1([H:42])[H:43])([H:14])[H:15].[O:3]([H:16])[H:44]" \
  --config scripts/conf/rits.yaml \
  --ckpt data/rits.ckpt \
  --output examples/organocatalysis_hajos/step12_complex_2.xyz \
  --kekulize \
  --add_stereo \
  --num_steps 25 \
  --n_samples 100

# Step 2: step2_central_cc
PYTHONPATH="./src" python scripts/sample_transition_state.py \
  --reaction_smarts "[C:1](=[C:2]([C:4]([C:5]([C:6]1([C:7]([H:21])([H:22])[H:23])[C:8](=[O:10])[C:9]([H:24])([H:25])[C:11]([H:26])([H:27])[C:12]1=[O:13])([H:19])[H:20])([H:17])[H:18])[N:35]1[C@@:31]([C:29](=[O:28])[O:30][H:36])([H:37])[C:32]([H:38])([H:39])[C:33]([H:40])([H:41])[C:34]1([H:42])[H:43])([H:14])[H:15]>>[C:1]1([H:14])([H:15])[C:2](=[N+:35]2[C@@:31]([C:29](=[O:28])[O-:30])([H:37])[C:32]([H:38])([H:39])[C:33]([H:40])([H:41])[C:34]2([H:42])[H:43])[C:4]([H:16])([H:18])[C:5]([H:19])([H:20])[C@:6]2([C:7]([H:21])([H:22])[H:23])[C:8](=[O:10])[C:9]([H:24])([H:25])[C:11]([H:26])([H:27])[C@:12]12[O:13][H:36]" \
  --config scripts/conf/rits.yaml \
  --ckpt data/rits.ckpt \
  --output examples/organocatalysis_hajos/step2_central_cc.xyz \
  --kekulize \
  --add_stereo \
  --num_steps 25 \
  --n_samples 100

# Step 3.1: step31_release_1
PYTHONPATH="./src" python scripts/sample_transition_state.py \
  --reaction_smarts "[C:1]1([H:14])([H:15])[C:2](=[N+:35]2[C@@:31]([C:29](=[O:28])[O-:30])([H:37])[C:32]([H:38])([H:39])[C:33]([H:40])([H:41])[C:34]2([H:42])[H:43])[C:4]([H:16])([H:18])[C:5]([H:19])([H:20])[C@:6]2([C:7]([H:21])([H:22])[H:23])[C:8](=[O:10])[C:9]([H:24])([H:25])[C:11]([H:26])([H:27])[C@:12]12[O:13][H:36].[O:45]([H:46])[H:47]>>[C:1]1([H:14])([H:15])[C:2]([N:35]2[C@@:31]([C:29](=[O:28])[O:30][H:47])([H:37])[C:32]([H:38])([H:39])[C:33]([H:40])([H:41])[C:34]2([H:42])[H:43])([O:45][H:46])[C:4]([H:16])([H:18])[C:5]([H:19])([H:20])[C@:6]2([C:7]([H:21])([H:22])[H:23])[C:8](=[O:10])[C:9]([H:24])([H:25])[C:11]([H:26])([H:27])[C@:12]12[O:13][H:36]" \
  --config scripts/conf/rits.yaml \
  --ckpt data/rits.ckpt \
  --output examples/organocatalysis_hajos/step31_release_1.xyz \
  --kekulize \
  --add_stereo \
  --num_steps 25 \
  --n_samples 100

# Step 3.2: step32_release_2
PYTHONPATH="./src" python scripts/sample_transition_state.py \
  --reaction_smarts "[C:1]1([H:14])([H:15])[C:2]([N:35]2[C@@:31]([C:29](=[O:28])[O:30][H:47])([H:37])[C:32]([H:38])([H:39])[C:33]([H:40])([H:41])[C:34]2([H:42])[H:43])([O:45][H:46])[C:4]([H:16])([H:18])[C:5]([H:19])([H:20])[C@:6]2([C:7]([H:21])([H:22])[H:23])[C:8](=[O:10])[C:9]([H:24])([H:25])[C:11]([H:26])([H:27])[C@:12]12[O:13][H:36]>>[C:1]1([H:14])([H:15])[C:2](=[O:45])[C:4]([H:16])([H:18])[C:5]([H:19])([H:20])[C@:6]2([C:7]([H:21])([H:22])[H:23])[C:8](=[O:10])[C:9]([H:24])([H:25])[C:11]([H:26])([H:27])[C@:12]12[O:13][H:36].[O:28]=[C:29]([O:30][H:47])[C@:31]1([H:37])[C:32]([H:38])([H:39])[C:33]([H:40])([H:41])[C:34]([H:42])([H:43])[N:35]1[H:46]" \
  --config scripts/conf/rits.yaml \
  --ckpt data/rits.ckpt \
  --output examples/organocatalysis_hajos/step32_release_2.xyz \
  --kekulize \
  --add_stereo \
  --num_steps 25 \
  --n_samples 100

# Extra: no-proline overall intramolecular cyclization (mapped, explicit H)
# PYTHONPATH="./src" python scripts/sample_transition_state.py \
#   --reaction_smarts "[C:1]([C:2](=[O:3])[C:4]([C:5]([C:6]1([C:7]([H:21])([H:22])[H:23])[C:8](=[O:10])[C:9]([H:24])([H:25])[C:11]([H:26])([H:27])[C:12]1=[O:13])([H:19])[H:20])([H:17])[H:18])([H:14])([H:15])[H:16]>>[C:1]1([H:14])([H:15])[C:2](=[O:3])[C:4]([H:17])([H:18])[C:5]([H:19])([H:20])[C:6]2([C:7]([H:21])([H:22])[H:23])[C:8](=[O:10])[C:9]([H:24])([H:25])[C:11]([H:26])([H:27])[C:12]12[O:13][H:16]" \
#   --config scripts/conf/rits.yaml \
#   --ckpt data/rits.ckpt \
#   --output examples/organocatalysis_hajos/no_proline_overall.xyz \
#   --kekulize \
#   --add_stereo \
#   --num_steps 25 \
#   --n_samples 100
