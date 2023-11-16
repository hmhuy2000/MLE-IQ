# CUDA_VISIBLE_DEVICES=3 python train_SQIL.py env=cheetah agent=sac \
# env.name=HalfCheetah-v3 \
# agent.init_temp=0.01 env.eval_interval=5e3 \
# env.demo=new_HalfCheetah-v3/0.hdf5 expert.demos=0 env.learn_steps=1e6 \
# env.sub_optimal_demo=[new_HalfCheetah-v3/3.hdf5,new_HalfCheetah-v3/2.hdf5,new_HalfCheetah-v3/1.hdf5,new_HalfCheetah-v3/0.hdf5] \
# env.num_sub_optimal_demo=[100,100,100,10] \
# agent.actor_lr=3e-05 seed=0 

#--------------------#

# CUDA_VISIBLE_DEVICES=2 python train_SQIL.py env=ant agent=sac \
# env.name=Ant-v3 \
# agent.init_temp=0.01 env.eval_interval=5e3 \
# env.demo=new_Ant-v3/0.hdf5 expert.demos=0 env.learn_steps=1e6 \
# env.sub_optimal_demo=[new_Ant-v3/3.hdf5,new_Ant-v3/2.hdf5,new_Ant-v3/1.hdf5,new_Ant-v3/0.hdf5] \
# env.num_sub_optimal_demo=[100,100,100,10] \
# agent.actor_lr=3e-05 seed=7

#--------------------#

# CUDA_VISIBLE_DEVICES=0 python train_SQIL.py env=walker2d agent=sac \
# env.name=Walker2d-v3 method.loss=value \
# agent.init_temp=0.01 env.eval_interval=5e3 \
# env.demo=new_Walker2d-v3/0.hdf5 expert.demos=0 env.learn_steps=1e6 \
# env.sub_optimal_demo=[new_Walker2d-v3/3.hdf5,new_Walker2d-v3/2.hdf5,new_Walker2d-v3/1.hdf5,new_Walker2d-v3/0.hdf5] \
# env.num_sub_optimal_demo=[100,100,100,10] \
# agent.actor_lr=3e-05 seed=1

#--------------------#

# CUDA_VISIBLE_DEVICES=1 python train_SQIL.py env=humanoid agent=sac \
# env.name=Humanoid-v3 method.loss=value \
# agent.init_temp=0.01 env.eval_interval=5e3 \
# env.demo=new_Humanoid-v3/0.hdf5 expert.demos=0 env.learn_steps=1e6 \
# env.sub_optimal_demo=[new_Humanoid-v3/3.hdf5,new_Humanoid-v3/2.hdf5,new_Humanoid-v3/1.hdf5,new_Humanoid-v3/0.hdf5] \
# env.num_sub_optimal_demo=[100,100,100,25] \
# agent.actor_lr=3e-05 seed=7

#--------------------#

# CUDA_VISIBLE_DEVICES=0 python train_SQIL.py env=hopper agent=sac \
# env.name=Hopper-v3 method.loss=value gamma=0.999 \
# agent.init_temp=0.01 env.eval_interval=5e3 \
# env.demo=new_Hopper-v3/0.hdf5 expert.demos=0 env.learn_steps=1e6 \
# env.sub_optimal_demo=[new_Hopper-v3/3.hdf5,new_Hopper-v3/2.hdf5,new_Hopper-v3/1.hdf5,new_Hopper-v3/0.hdf5] \
# env.num_sub_optimal_demo=[100,100,100,10] \
# agent.actor_lr=3e-05 seed=7