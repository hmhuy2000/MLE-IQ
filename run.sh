# CUDA_VISIBLE_DEVICES=3 python train.py env=cheetah agent=sac expert.demos=5 \
# method.loss=value method.regularize=True agent.actor_lr=3e-05 seed=0 

CUDA_VISIBLE_DEVICES=3 python train.py env=cheetah agent=sac expert.demos=25 \
method.loss=value_expert method.chi=True agent.actor_lr=3e-05 seed=0 \
offline=True agent.init_temp=0.01