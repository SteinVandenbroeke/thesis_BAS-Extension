cd ..
source .run_venv/bin/activate
python3 train_bas.py \
    --dataset coco \
    --coco_root ../../../datasets/coco_dataset/coco2017/ \
    --train_list ../../../datasets/coco_dataset/coco2017/train2017 \
    --val_list ../../../datasets/coco_dataset/coco2017/val2017 \
    --num_classes 80 \
    --batch_size 16 \
    --epochs 10 \
    --lr 0.005 \
    --arch resnet50_bas \
    --save_path sess
