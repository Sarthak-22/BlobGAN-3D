CUDA_VISIBLE_DEVICES=2,3 python src/run.py +experiment=[blobgan,local,jitter] wandb.name='Exp_3'    \
dataset.dataloader.batch_size=32 \
+model.log_fid_every_epoch=False \
dataset=multi_clevr \
+dataset.cameras=[Camera,Camera.001] \
+dataset.path=datasets/CLEVR_new    \
model.n_features=10                  \