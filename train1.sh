#!/bin/bash
model=topdown
rnn_size=1024
encoding_size=1024
num_layers=1
dropout=0.5
batch_size=50
gpu_num=1
seed=141
iters=1
att_num=100
epoch=300
tar_metric=CIDEr
old_id=val_1255
id=n1_cider_1
if [ ! -d "/data3/liuhan/WCST/log/$id"  ]; then
    mkdir /data3/liuhan/WCST/log/$id
fi
export PYTHONPATH=/home/liuhan/image-captioning/coco-caption/

    #--start_from ../caption_model/log/${old_id} --old_id ${old_id} \
CUDA_VISIBLE_DEVICES=0 python \
    -u train_wcst.py --id $id --caption_model $model --drop_prob_lm $dropout \
    --start_from  log/${old_id} --old_id ${old_id} \
    --annFile /data3/liuhan/dataset/coco/captions_trainval2014.json \
    --seed $seed \
    --optim adam \
    --dataset_ix_start -1 --dataset_ix_end -1 \
    --input_encoding_size ${encoding_size} \
    --use_img 0 --use_topic 0 --use_fc 1\
    --start_from_best 1 \
    --rnn_size $rnn_size --num_layers $num_layers \
    --input_json /data3/liuhan/dataset/coco/coco_meta2.json \
    --att_feat_num ${att_num} \
    --input_fc_dir /data3/liuhan/dataset/coco/bottom-up-data/fc_bottom_up \
    --input_att_dir /data3/liuhan/dataset/coco/bottom-up-data/att_bottom_up_10_100 \
    --input_label_h5 /data3/liuhan/dataset/coco/coco_label2.h5 \
    --cached_tokens /data3/liuhan/dataset/coco/coco-all2-idxs \
    --batch_size $batch_size --iter_times $iters --gpu_num $gpu_num \
    --fix_rnn 0 --learning_rate 5e-5 --learning_rate_decay_start -1 \
    --learning_rate_decay_every 4 --learning_rate_decay_rate 0.8 \
    --checkpoint_path /data3/liuhan/WCST/log/$id \
    --save_every 10 \
    --tar_metric ${tar_metric} \
    --use_scst 1 \
    --use_bos 0 \
    --use_eos 1 \
    --use_one_gram 0.2 \
    --alpha 0.5 --alpha_decay_start -1 --alpha_decay_every 4 --alpha_decay_rate 0.05 --alpha_max 0.5 \
    --gamma 0.5 \
    --val_images_use -1 --language_eval 1 --max_epochs $epoch  \
    2>&1 | tee /data3/liuhan/WCST/log/$id/train.log





