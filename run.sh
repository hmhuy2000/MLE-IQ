CUDA_VISIBLE_DEVICES=0 python train_MLE.py env=ant agent=sac \
agent.init_temp=0.01 \
env.demo=ant_expert-v2.hdf5 expert.demos=100 \
env.sub_optimal_demo=ant_medium-v2.hdf5 env.num_sub_optimal_demo=-1 \
method.loss=value method.regularize=True agent.actor_lr=3e-05 seed=0 \
embed.latent_dim=128