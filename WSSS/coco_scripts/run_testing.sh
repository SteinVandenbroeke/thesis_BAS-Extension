cd ..
source .run_venv/bin/activate
python3 run_sample.py \
    --dataset coco \
    --coco_root ../../../datasets/coco_dataset/coco2017/ \
    --train_list ../../../datasets/coco_dataset/coco2017/train2017 \
    --val_list ../../../datasets/coco_dataset/coco2017/val2017 \
    --infer_list ../../../datasets/coco_dataset/coco2017/train2017 \
    --num_classes 80 \
    --cam_network net.resnet50_bas \
    --cam_weights_name sess/resnet50_bas/epoch_10.pth.tar
