# CUDA_VISIBLE_DEVICES=0 python train_both.py env=cheetah agent=sac \
# env.name=HalfCheetah-v2 method.loss=value \
# agent.init_temp=0.03 env.eval_interval=1e4 \
# env.demo=halfcheetah_expert-v2.hdf5 expert.demos=0 env.learn_steps=1e6 \
# env.sub_optimal_demo=[halfcheetah_random-v2.hdf5,halfcheetah_medium-v2.hdf5,halfcheetah_expert-v2.hdf5] \
# env.num_sub_optimal_demo=[200000,100000,10000] \
# expert.reward_arr=[0.0,0.5,1.0] \
# agent.actor_lr=3e-05 seed=0

# CUDA_VISIBLE_DEVICES=3 python train_both.py env=ant agent=sac \
# env.name=Ant-v2 method.loss=value \
# agent.init_temp=0.003 env.eval_interval=1e4 \
# env.demo=ant_expert-v2.hdf5 expert.demos=0 env.learn_steps=1e6 \
# env.sub_optimal_demo=[ant_random-v2.hdf5,ant_medium-v2.hdf5,ant_expert-v2.hdf5] \
# env.num_sub_optimal_demo=[1000000,1000000,10000] \
# expert.reward_arr=[0.0,0.5,1.0] \
# agent.actor_lr=3e-05 seed=0