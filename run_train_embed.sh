CUDA_VISIBLE_DEVICES=2 python train_embed.py env=ant agent=sac \
embed.num_demos=-1 \
embed.random_dataset=ant_random-v2.hdf5 \
embed.train_dataset=ant_medium-v2.hdf5 \
embed.eval_dataset=ant_expert-v2.hdf5 \
embed.latent_dim=32 embed.lr=0.00003

# CUDA_VISIBLE_DEVICES=0 python train_embed.py env=cheetah agent=sac \
# embed.num_demos=-1 \
# embed.random_dataset=halfcheetah_random-v2.hdf5 \
# embed.train_dataset=halfcheetah_medium-v2.hdf5 \
# embed.eval_dataset=halfcheetah_expert-v2.hdf5 \
# embed.latent_dim=32 \
