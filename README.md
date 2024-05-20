# DCorr-Nav
DCorr-Nav: Bridging Perception and Action with Direction-Aware Correlation for Image-Goal Navigation

## Abstract
Recent works in image-goal navigation typically involve learning a perception-to-action navigation policy: they first capture semantic features of the goal image and egocentric image independently and then pass them to the policy network for action prediction. Despite a remarkable series of work efforts on visual representations, there are some challenges in terms of navigation efficiency and robustness for these semantic feature-dependent methods. In this paper, we are working to address these challenges by proposing a direction-aware correlation-dependent method~(DCorr-Nav). Specifically, we construct correlations between the features extracted in the perception step and pass the correlation information to the policy network, i.e., training a perception-correlation-action policy. We gradually reinforce the construction of correlations with three versions and eventually developing a powerful DCorr-Nav. Extensive evaluation of the DCorr-Nav on 3 benchmark datasets~(Gibson, HM3D, and MP3D) shows the superior performance in terms of navigation efficiency~(SPL). And it significantly outperforms previous state-of-the-art methods across all metrics~(SPL, SR) under the “user-matched goal” setting, showing potential for real-world applications.
![](figs/methods.png)

## Install
### Installing on the host machine
```bash
# create conda env
conda create -n DCorr-Nav python=3.8
conda activate DCorr-Nav
conda install habitat-sim=0.2.2 withbullet headless -c conda-forge -c aihabitat
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

git submodule init
git submodule update
cd habitat-lab
git checkout 1f7cfbdd3debc825f1f2fd4b9e1a8d6d4bc9bfc7
pip install -e habitat-lab 
pip install -e habitat-baselines
cd ..
pip install requirements.txt
```

## Data preparation
<!-- 
| ObjectNav   |   Gibson     | train    |  [objectnav_gibson_train](https://utexas.box.com/s/7qtqqkxa37l969qrkwdn0lkwitmyropp)    | `./data/datasets/zer/objectnav/gibson/v1/` |
| ObjectNav   |   Gibson     | val    |  [objectnav_gibson_val](https://utexas.box.com/s/wu28ms025o83ii4mwfljot1soj5dc7qo)    | `./data/datasets/zer/objectnav/gibson/v1/` | -->

### Datasets
Follow the [official guidance](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#gibson-and-3dscenegraph-datasets) to download `Gibson`, `HM3D`, and `MP3D` scene datas. And follow FGPrompt download the dataset.zip. 
```
data/datasets/
└── imagenav
    ├── gibson
    ├── hm3d
    └── mp3d
data/scene_datasets
    ├── gibson
    ├── hm3d
    └── mp3d
```

## Training
```bash
MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet python -m torch.distributed.launch 
--nproc_per_node=4 
--master_port=11200 
--nnodes=1 
--node_rank=0 
--master_addr=127.0.0.1 
run.py 
--overwrite 
--exp-config 
exp_config/ddppo_imagenav_gibson.yaml,policy,reward,dataset,sensors,DCorr-Nav 
--run-type 
train 
--model-dir 
results/imagenav/exp1
```

## Evaluation: Reproduce the Results in the main paper️
### Evaluation on Gibson

```bash
# agent-matched setting
MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet python -m torch.distributed.launch 
--nproc_per_node=1 
--master_port=11200 
--nnodes=1 
--node_rank=0 
--master_addr=127.0.0.1 
run.py 
--exp-config
exp_config/ddppo_imagenav_gibson.yaml,policy,reward,dataset,sensors,DCorr-Nav,eval
--run-type
eval
--model-dir
results/imagenav/exp2
habitat_baselines.eval_ckpt_path_dir
/checkpoint/checkpoint.pth

# user-matched setting
MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet python -m torch.distributed.launch 
--nproc_per_node=1 
--master_port=11200 
--nnodes=1 
--node_rank=0 
--master_addr=127.0.0.1 
run.py 
--exp-config
exp_config/ddppo_imagenav_gibson.yaml,policy,reward,dataset,sensors,DCorr-Nav,eval
--run-type
eval
--model-dir
results/imagenav/exp2
habitat_baselines.eval_ckpt_path_dir
/checkpoint/checkpoint.pth
--habitat.task.imagegoal_sensor_v2.augmentation.activate
True
```




### Cross-domain Evaluation on HM3D
```bash
# agent-matched setting
MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet python -m torch.distributed.launch 
--nproc_per_node=1 
--master_port=11200 
--nnodes=1 
--node_rank=0 
--master_addr=127.0.0.1 
run.py 
--exp-config 
exp_config/ddppo_imagenav_gibson.yaml,policy,reward,dataset-hm3d,sensors,DCorr-Nav,eval 
--run-type 
eval 
--model-dir 
results/imagenav/exp2
habitat_baselines.eval_ckpt_path_dir 
/checkpoint/checkpoint.pth
habitat_baselines.eval.split 
val_easy # [val_easy, val_hard, val_medium]

# user-matched setting
MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet python -m torch.distributed.launch 
--nproc_per_node=1 
--master_port=11200 
--nnodes=1 
--node_rank=0 
--master_addr=127.0.0.1 
run.py 
--exp-config 
exp_config/ddppo_imagenav_gibson.yaml,policy,reward,dataset-hm3d,sensors,DCorr-Nav,eval 
--run-type 
eval 
--model-dir 
results/imagenav/exp2
habitat_baselines.eval_ckpt_path_dir 
/checkpoint/checkpoint.pth
habitat_baselines.eval.split 
val_easy # [val_easy, val_hard, val_medium]
--habitat.task.imagegoal_sensor_v2.augmentation.activate
True
```

### Cross-domain Evaluation on MP3D
```bash
# agent-matched setting
MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet python -m torch.distributed.launch 
--nproc_per_node=1 
--master_port=11200 
--nnodes=1 
--node_rank=0 
--master_addr=127.0.0.1 
run.py 
--exp-config 
exp_config/ddppo_imagenav_gibson.yaml,policy,reward,dataset-mp3d,sensors,DCorr-Nav,eval
--run-type 
eval 
--model-dir 
results/imagenav/exp2
habitat_baselines.eval_ckpt_path_dir 
/checkpoint/checkpoint.pth
habitat_baselines.eval.split 
test_easy # [test_easy, test_hard, test_medium]

# user-matched setting
MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet python -m torch.distributed.launch 
--nproc_per_node=1 
--master_port=11200 
--nnodes=1 
--node_rank=0 
--master_addr=127.0.0.1 
run.py 
--exp-config 
exp_config/ddppo_imagenav_gibson.yaml,policy,reward,dataset-mp3d,sensors,DCorr-Nav,eval
--run-type 
eval 
--model-dir 
results/imagenav/exp2
habitat_baselines.eval_ckpt_path_dir 
/checkpoint/checkpoint.pth
habitat_baselines.eval.split 
test_easy # [test_easy, test_hard, test_medium]
--habitat.task.imagegoal_sensor_v2.augmentation.activate
True
```
