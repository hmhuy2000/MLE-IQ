# CUDA_VISIBLE_DEVICES=0 python train_both.py env=cheetah agent=sac \
# env.name=HalfCheetah-v3 method.loss=strict_value \
# agent.init_temp=0.03 env.eval_interval=1e4 \
# env.demo=final_HalfCheetah-v3/0.hdf5 expert.demos=0 env.learn_steps=1e6 \
# env.sub_optimal_demo=[final_HalfCheetah-v3/4.hdf5,final_HalfCheetah-v3/3.hdf5,final_HalfCheetah-v3/2.hdf5,final_HalfCheetah-v3/1.hdf5,final_HalfCheetah-v3/0.hdf5] \
# env.num_sub_optimal_demo=[100000,50000,20000,10000,5000] \
# expert.reward_arr=[0.3,0.6,0.9,1.2,1.5] \
# agent.actor_lr=3e-05 seed=1

# CUDA_VISIBLE_DEVICES=1 python train_both.py env=ant agent=sac \
# env.name=Ant-v3 method.loss=strict_value \
# agent.init_temp=0.003 env.eval_interval=5e3 \
# env.demo=final_Ant-v3/0.hdf5 expert.demos=0 env.learn_steps=1e6 \
# env.sub_optimal_demo=[final_Ant-v3/4.hdf5,final_Ant-v3/3.hdf5,final_Ant-v3/2.hdf5,final_Ant-v3/1.hdf5,final_Ant-v3/0.hdf5] \
# env.num_sub_optimal_demo=[100000,50000,20000,10000,5000] \
# expert.reward_arr=[0.3,0.6,0.9,1.2,1.5] \
# agent.actor_lr=3e-05 seed=1

# CUDA_VISIBLE_DEVICES=2 python train_both.py env=humanoid agent=sac \
# env.name=Humanoid-v3 method.loss=strict_value \
# agent.init_temp=0.01 env.eval_interval=1e4 \
# env.demo=final_Humanoid-v3/0.hdf5 expert.demos=0 env.learn_steps=1e6 \
# env.sub_optimal_demo=[final_Humanoid-v3/4.hdf5,final_Humanoid-v3/3.hdf5,final_Humanoid-v3/2.hdf5,final_Humanoid-v3/1.hdf5,final_Humanoid-v3/0.hdf5] \
# env.num_sub_optimal_demo=[100000,50000,20000,10000,5000] \
# expert.reward_arr=[0.3,0.6,0.9,1.2,1.5] \
# agent.actor_lr=3e-5 seed=0

# CUDA_VISIBLE_DEVICES=3 python train_both.py env=walker2d agent=sac \
# env.name=Walker2d-v3 method.loss=strict_value \
# agent.init_temp=0.03 env.eval_interval=1e4 \
# env.demo=final_Walker2d-v3/0.hdf5 expert.demos=0 env.learn_steps=1e6 \
# env.sub_optimal_demo=[final_Walker2d-v3/4.hdf5,final_Walker2d-v3/3.hdf5,final_Walker2d-v3/2.hdf5,final_Walker2d-v3/1.hdf5,final_Walker2d-v3/0.hdf5] \
# env.num_sub_optimal_demo=[100000,50000,20000,10000,5000] \
# expert.reward_arr=[0.3,0.6,0.9,1.2,1.5] \
# agent.actor_lr=3e-05 seed=0

# CUDA_VISIBLE_DEVICES=3 python train_both.py env=hopper agent=sac \
# env.name=Hopper-v3 method.loss=value \
# agent.init_temp=0.03 env.eval_interval=5e3 \
# env.demo=final_Hopper-v3/0.hdf5 expert.demos=0 env.learn_steps=3e6 \
# env.sub_optimal_demo=[final_Hopper-v3/4.hdf5,final_Hopper-v3/3.hdf5,final_Hopper-v3/2.hdf5,final_Hopper-v3/0.hdf5] \
# env.num_sub_optimal_demo=[100000,100000,100000,1000] \
# expert.reward_arr=[0.3,0.6,0.9,1.5] \
# agent.actor_lr=3e-05 seed=0