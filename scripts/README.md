## Scripts Manual

You should command below before launch scripts.
```bash
make init
```

### 1. Single Step Trainer(SAC+HER)

```bash
python single_step_train.py --env-id PandaReach-v1

#usage: single_step_train.py [-h] [--env-id ENV_ID]

#Single-step RL Trainer using SAC+HER

#optional arguments:
#  -h, --help       show this help message and exit
#  --env-id ENV_ID  panda-gym environment[PandaReach-v1, PandaSlice-v1, PandaPush-v1, PandaPickAndPlace-v1, PandaStack-v1]

```

### 2. Multi Step Trainer(HAC)

```bash
python multi_step_train.py --env-id PandaStack-v1

#usage: multi_step_train.py [-h] [--env-id ENV_ID] [--no-render NO_RENDER]

#Multi-step RL Trainer Arguments using HAC

#optional arguments:
#  -h, --help            show this help message and exit
#  --env-id ENV_ID       Environment default: PandaStack-v1, option :[PandaReach-v1(not allowed), PandaSlice-v1(not allowed),
#                        PandaPush-v1(not allowed), PandaPickAndPlace-v1(not allowed), PandaStack-v1]
#  --no-render NO_RENDER
#                        gym render option

```

### Supporting TensorBoard
```bash
tensorboard --logdir=$(LOGDIR)
# e.g. `tensorboard --logdir=pretrained/PandaStack-v1/logs/`
```