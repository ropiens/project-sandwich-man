## Scripts Manual

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
python multi_step_train.py --env-id PandaReach-v1

#usage: multi_step_train.py [-h] [--env-id ENV_ID]

#Multi-step RL Trainer using HAC

#optional arguments:
#  -h, --help       show this help message and exit
#  --env-id ENV_ID  panda-gym environment[PandaReach-v1, PandaSlice-v1, PandaPush-v1, PandaPickAndPlace-v1, PandaStack-v1]

```
