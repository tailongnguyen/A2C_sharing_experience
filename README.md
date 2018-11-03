# A2C_sharing_experience
Multi-task learning with Advantage Actor Critic  and sharing experience 

## Transfer experience from pretrained agents

Use pretrained value function and policy to estimate advantages and importance weight in shared states, respectively. We obviously do not re-train the pretrained agents.
Specify the pretrained task by hardcoding the code (for now).
For example,
```
pretrained_dir = ["logs/2018-11-01_21-12-43_test_save_model/num_task_1-num_episode_12-num_iters_50-lr_0.005-use_gae/checkpoints/"]
if args.transfer:
	pretrained = [0]
else:
	pretrained = []
```

means that we use pretrained agent of task `0` and the corresponding weights path is `"logs/2018-11-01_21-12-43_test_save_model/num_task_1-num_episode_12-num_iters_50-lr_0.005-use_gae/checkpoints/"`.


