export CUDA_VISIBLE_DEVICES=0
model_path="./model/faster/epoch_12.pth"
python tools/test.py ./outputs/faster/faster-rcnn_r50_fpn_1x_coco_lzf.py  $model_path --out "./output_test/faster.pkl" --work-dir="./output_test"
