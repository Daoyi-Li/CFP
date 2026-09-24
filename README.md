# CFP

## Get Started
This project requires **Python 3.10** and CUDA 12.1. We recommend using Conda:
### 1. Create Conda Environment
```bash
conda create -n cfp python=3.10 -y
conda activate cfp
```

### 2. Install PyTorch (CUDA 12.1)
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install JAX (CUDA 12)
If your training utilizes JAX / Flax components:
```bash
pip install -U "jax[cuda12]"
```

### 4. Install Main Dependencies
```bash
pip install -r requirements.txt
```

### 5. Benchmark / Simulator Environments (Robosuite & OGBench)
If you evaluate on Robosuite or OGBench tasks, make sure their assets and MuJoCo are installed:
```bash
# OGBench (if needed)
pip install ogbench==1.1.2
```

## Run
We provide pretrained actor in `Supplementary Material`. You can simply reuse them!

If you want to run the pretrained actor by yourself, please run
``` bash
python main_save_para.py --run_group=any_network_emb --agent=agents/acfql.py   --env_name=cube-triple-play-singletask-task1-v0 --sparse=False --horizon_length=5
```
This will save the pretrained actor once the offline training is done.

Please note that since three of the four algorithms we use are implemented in `agent/acfql.py`, you should rename the paths in the main program when running different algorithms to avoid path conflicts.

### QC
#### O2O
``` bash
python main_o2o.py --run_group=reproduce --agent.alpha=100 --discount=0.99 --env_name=cube-triple-play-singletask-task1-v0 --sparse=False --horizon_length=5 --agent.actor_type=best-of-n --agent.actor_num_samples=32 
```
#### CFP
``` bash
python main_warmup.py --run_group=any_network_emb --agent=agents/acfql.py --agent.alpha=100 --env_name=cube-triple-play-singletask-task1-v0 --sparse=False --horizon_length=5 --skip_offline_training=True --critic_warmup_steps=10000 --load_actor_checkpoint=/Your/Path/To/CFP/pretrained_actors/FQL/cube-triple-play-singletask-task1-v0/actor_checkpoints/actor_step_1000000.flax --agent.actor_type=best-of-n --agent.actor_num_samples=32
```

### FQL
#### O2O
``` bash
python main_o2o.py --run_group=reproduce --agent.alpha=100 --discount=0.99 --env_name=cube-triple-play-singletask-task1-v0 --sparse=False --horizon_length=1  
```
#### CFP
``` bash
python main_warmup.py --run_group=any_network_emb --agent=agents/acfql.py --agent.alpha=100 --env_name=cube-triple-play-singletask-task1-v0 --sparse=False --horizon_length=1 --skip_offline_training=True --critic_warmup_steps=10000 --load_actor_checkpoint=/Your/Path/To/CFP/pretrained_actors/FQL/cube-triple-play-singletask-task1-v0/actor_checkpoints/actor_step_1000000.flax
```

### QCFQL
#### O2O
``` bash
python main_o2o.py --run_group=reproduce --agent.alpha=100 --discount=0.99 --env_name=cube-triple-play-singletask-task1-v0 --sparse=False --horizon_length=5  
```
#### CFP
``` bash
python main_warmup.py --run_group=any_network_emb --agent=agents/acfql.py --agent.alpha=100 --env_name=cube-triple-play-singletask-task1-v0 --sparse=False --horizon_length=5 --skip_offline_training=True --critic_warmup_steps=10000 --load_actor_checkpoint=/Your/Path/To/CFP/pretrained_actors/QCFQL/cube-triple-play-singletask-task1-v0/actor_checkpoints/actor_step_1000000.flax
```


### QCFQL-nstep
#### O2O
``` bash
python main_o2o.py --run_group=reproduce --agent=agents/acfql_nstep.py --agent.alpha=100 --discount=0.99 --env_name=cube-triple-play-singletask-task1-v0 --sparse=False --horizon_length=5  
```
#### CFP
``` bash
python main_warmup.py --run_group=any_network_emb --agent=agents/acfql_nstep.py --agent.alpha=100 --env_name=cube-triple-play-singletask-task1-v0 --sparse=False --horizon_length=5 --skip_offline_training=True --critic_warmup_steps=10000 --load_actor_checkpoint=/Your/Path/To/CFP/pretrained_actors/QCFQL-nstep/cube-triple-play-singletask-task1-v0/actor_checkpoints/actor_step_1000000.flax
```

## Check list
We list below some parameter configurations that may cause issues. Please pay attention to them when implementing the code.

### QC
Remember to add __--agent.actor_type=best-of-n --agent.actor_num_samples=32__

### FQL
Remember to confirm that __--horizon_length=1__

### QCFQL-nstep
Remember to confirm that __--agent=agents/acfql_nstep.py__

---

We also list some potential issues related with domains.
### Scene
Remember to confirm that __--sparse=True__

### Cube Quadruple
Remember to add __--ogbench_dataset_dir=Real/Path/To/Your/cube-quadruple-play-100m-v0/__
to make sure that the code is using the 100M dataset.

## Dataset
For OGBench, we use the default dataset (except for Cube Quadruple). Generally, the corresponding dataset is automatically downloaded when the code is run for the first time. Please ensure you are using the "play" and "singletask" versions. 
If you want to download the dataset mannuly, please visit https://rail.eecs.berkeley.edu/datasets/ogbench/.

Make sure that you download the right 100M dataset for Cube Quadruple.

For Robomimic, you can downloead it from https://robomimic.github.io/docs/datasets/robomimic_v0.1.html.

**All websites are provided officially. They do not compromise anonymity in any way.**

