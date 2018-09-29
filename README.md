# A2C_sharing_experience
Multi-task learning with Advantage Actor Critic  and sharing experience 

## To run the code

- To start training the agents, run: `python train.py <args_go_here>`
- You can make a custom map by adding it into `env/map.py` and modifying the `env/sxsy.py` (init states when training) using `generate_start_positions.py`

## About arguments

- See in `train.py`
- Arguments setting recommendations: 
	- `num_iters` between 5 and 20, otherwise the training will be slowed down
	- `num_epochs` should converge between 5k and 20k
	- `num_episodes` between 10 and 20



