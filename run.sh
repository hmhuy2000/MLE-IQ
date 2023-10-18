# CUDA_VISIBLE_DEVICES=3 python train.py env=ant agent=sac expert.demos=25 \
# agent.init_temp=0.01 env.demo=ant_expert-v2.hdf5 \
# method.loss=value method.regularize=True agent.actor_lr=3e-05 seed=1

# CUDA_VISIBLE_DEVICES=1 python train.py env=cheetah agent=sac expert.demos=1000 \
# offline=True agent.init_temp=0.01 env.demo=halfcheetah_expert-v2.hdf5 \
# method.loss=value_expert method.chi=True agent.actor_lr=3e-05 seed=0 

CUDA_VISIBLE_DEVICES=1 python train_MLE.py env=ant agent=sac \
agent.init_temp=0.01 \
env.demo=ant_expert-v2.hdf5 expert.demos=100 \
env.sub_optimal_demo=ant_medium-v2.hdf5 env.num_sub_optimal_demo=-1 \
method.loss=value method.regularize=True agent.actor_lr=3e-05 seed=0 \
embed.latent_dim=32 