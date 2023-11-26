CUDA_VISIBLE_DEVICES=1 python train_IQ.py env=cheetah agent=sac \
env.name=HalfCheetah-v3 \
agent.init_temp=0.01 env.eval_interval=5e3 \
env.demo=final_HalfCheetah-v3/0.hdf5 expert.demos=0 env.learn_steps=1e6 \
env.sub_optimal_demo=[final_HalfCheetah-v3/3.hdf5,final_HalfCheetah-v3/2.hdf5,final_HalfCheetah-v3/1.hdf5,final_HalfCheetah-v3/0.hdf5] \
env.num_sub_optimal_demo=[100000,50000,10000,5000] \
expert.reward_arr=[0.4,0.6,0.8,1.0] \
agent.reward_train_step=1e5 \
agent.actor_lr=3e-05 seed=0 
