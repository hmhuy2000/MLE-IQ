# CUDA_VISIBLE_DEVICES=2 python train.py env=ant agent=sac expert.demos=10 \
# agent.init_temp=0.001 env.demo=Ant-v2/7500.hdf5 \
# method.loss=value method.regularize=True agent.actor_lr=3e-05 seed=0 


# CUDA_VISIBLE_DEVICES=0 python train.py env=cheetah agent=sac expert.demos=5 \
# agent.init_temp=0.01 env.demo=12000.hdf5 \
# method.loss=value method.regularize=True agent.actor_lr=3e-05 seed=0 

# CUDA_VISIBLE_DEVICES=2 python train_MLE.py env=cheetah agent=sac \
# agent.init_temp=0.01 \
# env.demo=halfcheetah_expert-v2.hdf5 expert.demos=5 \
# env.sub_optimal_demo=halfcheetah_medium-v2.hdf5 env.num_sub_optimal_demo=50 \
# method.loss=value method.regularize=True agent.actor_lr=3e-05 seed=0 