# CUDA_VISIBLE_DEVICES=0 python new_train_reward.py env=cheetah agent=sac \
# env.name=HalfCheetah-v3 \
# agent.init_temp=0.01 env.eval_interval=5e3 \
# env.demo=final_HalfCheetah-v3/0.hdf5 expert.demos=0 env.learn_steps=1e6 \
# env.sub_optimal_demo=[final_HalfCheetah-v3/4.hdf5,final_HalfCheetah-v3/3.hdf5,final_HalfCheetah-v3/2.hdf5,final_HalfCheetah-v3/1.hdf5,final_HalfCheetah-v3/0.hdf5] \
# env.num_sub_optimal_demo=[100000,50000,20000,10000,5000] \
# agent.actor_lr=3e-05 seed=0 

# CUDA_VISIBLE_DEVICES=3 python new_train_reward.py env=ant agent=sac \
# env.name=Ant-v3 \
# agent.init_temp=0.001 env.eval_interval=5e3 \
# env.demo=final_Ant-v3/0.hdf5 expert.demos=0 env.learn_steps=1e6 \
# env.sub_optimal_demo=[final_Ant-v3/4.hdf5,final_Ant-v3/3.hdf5,final_Ant-v3/2.hdf5,final_Ant-v3/1.hdf5,final_Ant-v3/0.hdf5] \
# env.num_sub_optimal_demo=[100000,50000,20000,10000,5000] \
# agent.actor_lr=3e-05 seed=0

# CUDA_VISIBLE_DEVICES=1 python new_train_reward.py env=humanoid agent=sac \
# env.name=Humanoid-v3 method.loss=value \
# agent.init_temp=0.001 env.eval_interval=5e3 \
# env.demo=final_Humanoid-v3/0.hdf5 expert.demos=0 env.learn_steps=1e6 \
# env.sub_optimal_demo=[final_Humanoid-v3/4.hdf5,final_Humanoid-v3/3.hdf5,final_Humanoid-v3/2.hdf5,final_Humanoid-v3/1.hdf5,final_Humanoid-v3/0.hdf5] \
# env.num_sub_optimal_demo=[100000,100000,50000,20000,5000] \
# agent.actor_lr=3e-5 seed=0

# CUDA_VISIBLE_DEVICES=3 python new_train_reward.py env=hopper agent=sac \
# env.name=Hopper-v3 method.loss=value \
# agent.init_temp=0.001 env.eval_interval=5e3 \
# env.demo=final_Hopper-v3/0.hdf5 expert.demos=0 env.learn_steps=3e6 \
# env.sub_optimal_demo=[final_Hopper-v3/4.hdf5,final_Hopper-v3/3.hdf5,final_Hopper-v3/2.hdf5,final_Hopper-v3/1.hdf5,final_Hopper-v3/0.hdf5] \
# env.num_sub_optimal_demo=[250,250,10] \
# expert.reward_arr=[0.2,0.8,1.0] \
# agent.actor_lr=3e-05 seed=7