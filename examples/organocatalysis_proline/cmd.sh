# Proline-catalyzed aldol (step3 split into 2 substages; mapped shared IDs)

# Step 1.1: N->O proton transfer
PYTHONPATH="./src" python scripts/sample_transition_state.py \
  --reaction_smarts "[C:4](-[H:8])(-[H:9])(-[H:10])-[C:2](=[O:3])-[C:1](-[H:5])(-[H:6])-[H:7].[O:11]=[C:12](-[O:13]-[H:19])-[C@:14]1(-[H:20])-[C:15](-[H:21])(-[H:22])-[C:16](-[H:23])(-[H:24])-[C:17](-[H:25])(-[H:26])-[N:18]-1-[H:27]>>[O:11]=[C:12](-[O:13]-[H:19])-[C@:14]1(-[H:20])-[C:15](-[H:21])(-[H:22])-[C:16](-[H:23])(-[H:24])-[C:17](-[H:25])(-[H:26])-[N:18]-1-[C:2](-[O:3]-[H:27])(-[C:1](-[H:5])(-[H:6])-[H:7])-[C:4](-[H:8])(-[H:9])-[H:10]" \
  --config scripts/conf/rits.yaml \
  --ckpt data/rits.ckpt \
  --output examples/organocatalysis_proline/step11_proton_transfer.xyz \
  --kekulize \
  --add_stereo \
  --num_steps 25 \
  --n_samples 100

# Step 1.2: alpha-H transfer + water formation + enamine
PYTHONPATH="./src" python scripts/sample_transition_state.py \
  --reaction_smarts "[O:11]=[C:12](-[O:13]-[H:19])-[C@:14]1(-[H:20])-[C:15](-[H:21])(-[H:22])-[C:16](-[H:23])(-[H:24])-[C:17](-[H:25])(-[H:26])-[N:18]-1-[C:2](-[O:3]-[H:27])(-[C:1](-[H:5])(-[H:6])-[H:7])-[C:4](-[H:8])(-[H:9])-[H:10]>>[C:4](-[H:8])(-[H:9])(-[H:10])-[C:2](=[C:1](-[H:5])-[H:7])-[N:18]1-[C:17](-[H:25])(-[H:26])-[C:16](-[H:23])(-[H:24])-[C:15](-[H:21])(-[H:22])-[C@:14]-1(-[C:12](=[O:11])-[O:13]-[H:19])-[H:20].[O:3](-[H:6])-[H:27]" \
  --config scripts/conf/rits.yaml \
  --ckpt data/rits.ckpt \
  --output examples/organocatalysis_proline/step12_enamine_formation.xyz \
  --kekulize \
  --add_stereo \
  --num_steps 25 \
  --n_samples 100

# Step 2: Enamine + aldehyde -> C-C bond / iminium adduct
PYTHONPATH="./src" python scripts/sample_transition_state.py \
  --reaction_smarts "[C:1](=[C:2]([N:3]1[C:4]([H:14])([H:15])[C:5]([H:16])([H:17])[C:6]([H:18])([H:19])[C@:7]1([C:8](=[O:9])[O:10][H:21])[H:20])[C:11]([H:22])([H:23])[H:24])([H:12])[H:13].[C:25]([C:26]([C:27]([H:34])([H:35])[H:36])([C:28](=[O:29])[H:37])[H:33])([H:30])([H:31])[H:32]>>[C:1]([H:12])([H:13])([C@:28]([O:29][H:21])([H:37])[C:26]([C:27]([H:34])([H:35])[H:36])([H:33])[C:25]([H:30])([H:31])[H:32])-[C:2](=[N+:3]1[C:4]([H:14])([H:15])[C:5]([H:16])([H:17])[C:6]([H:18])([H:19])[C@:7]1([C:8](=[O:9])[O-:10])[H:20])[C:11]([H:22])([H:23])[H:24]" \
  --config scripts/conf/rits.yaml \
  --ckpt data/rits.ckpt \
  --output examples/organocatalysis_proline/step2_cc_bond_iminium.xyz \
  --kekulize \
  --add_stereo \
  --num_steps 25 \
  --n_samples 100

# Step 3.1: water adds to C2=N3; OH to C2 and H to O10
PYTHONPATH="./src" python scripts/sample_transition_state.py \
  --reaction_smarts "[C:1]([C:2](=[N+:3]1[C:4]([H:14])([H:15])[C:5]([H:16])([H:17])[C:6]([H:18])([H:19])[C@:7]1([C:8](=[O:9])[O-:10])[H:20])[C:11]([H:22])([H:23])[H:24])([H:12])([H:13])([C@:28]([O:29][H:21])([H:37])[C:26]([C:27]([H:34])([H:35])[H:36])([H:33])[C:25]([H:30])([H:31])[H:32]).[O:38]([H:39])[H:40]>>[C:1]([C:2]([O:38][H:39])(-[N:3]1[C:4]([H:14])([H:15])[C:5]([H:16])([H:17])[C:6]([H:18])([H:19])[C@:7]1([C:8](=[O:9])[O:10][H:40])[H:20])[C:11]([H:22])([H:23])[H:24])([H:12])([H:13])([C@:28]([O:29][H:21])([H:37])[C:26]([C:27]([H:34])([H:35])[H:36])([H:33])[C:25]([H:30])([H:31])[H:32])" \
  --config scripts/conf/rits.yaml \
  --ckpt data/rits.ckpt \
  --output examples/organocatalysis_proline/step31_water_addition.xyz \
  --kekulize \
  --add_stereo \
  --num_steps 25 \
  --n_samples 100

# Step 3.2: proton from O38 transfers to N3; C2-N3 breaks and C2=O38 forms
PYTHONPATH="./src" python scripts/sample_transition_state.py \
  --reaction_smarts "[C:1]([C:2]([O:38][H:39])(-[N:3]1[C:4]([H:14])([H:15])[C:5]([H:16])([H:17])[C:6]([H:18])([H:19])[C@:7]1([C:8](=[O:9])[O:10][H:40])[H:20])[C:11]([H:22])([H:23])[H:24])([H:12])([H:13])([C@:28]([O:29][H:21])([H:37])[C:26]([C:27]([H:34])([H:35])[H:36])([H:33])[C:25]([H:30])([H:31])[H:32])>>[C:1]([H:12])([H:13])([C@:28]([O:29][H:21])([H:37])[C:26]([C:27]([H:34])([H:35])[H:36])([H:33])[C:25]([H:30])([H:31])[H:32])-[C:2](=[O:38])[C:11]([H:22])([H:23])[H:24].[N:3]1([H:39])[C:4]([H:14])([H:15])[C:5]([H:16])([H:17])[C:6]([H:18])([H:19])[C@:7]1([C:8](=[O:9])[O:10][H:40])[H:20]" \
  --config scripts/conf/rits.yaml \
  --ckpt data/rits.ckpt \
  --output examples/organocatalysis_proline/step32_collapse_release.xyz \
  --kekulize \
  --add_stereo \
  --num_steps 25 \
  --n_samples 100
