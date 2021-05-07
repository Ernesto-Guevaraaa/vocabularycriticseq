#!/bin/bash
model=topdown
rnn_size=1024
encoding_size=1024
num_layers=1
batch_size=50
gpu_num=1
iters=1
att_num=100
beam_size=1
old_id=val_1255
id=test0
if [ ! -d "log/$id"  ]; then
    mkdir log/$id
fi
export PYTHONPATH=/home/liuhan/image-captioning/coco-caption

    #--start_from ../caption_model/log/${old_id} --old_id ${old_id} \
CUDA_VISIBLE_DEVICES=2 python \
    -u test.py --id $id --caption_model $model\
    --start_from log/${old_id} --old_id ${old_id} \
    --annFile data/captions_trainval2014.json \
    --input_encoding_size ${encoding_size} \
    --use_img 0 --use_topic 0 --use_fc 1\
    --start_from_best 1 \
    --rnn_size $rnn_size --num_layers $num_layers \
    --input_json data/coco_meta2.json \
    --img_fold data/images --img_size 512 --img_csize 448 \
    --att_feat_num ${att_num} \
    --input_fc_dir data/coco/bottom-up-data/fc_bottom_up \
    --input_att_dir data/coco/bottom-up-data/att_bottom_up_10_100 \
    --input_label_h5 data/coco/coco_label2.h5 \
    --batch_size $batch_size --iter_times $iters --gpu_num $gpu_num \
    --seq_per_img 1 \
    --test_split test --val_images_use 5000 --language_eval 1 \
    --dataset_ix_start -1 --dataset_ix_end -1 \
    --beam_size ${beam_size} --sample_n 1 \
    --invert 0 \
    2>&1 | tee log/$id/test.log





