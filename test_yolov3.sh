export CUDA_VISIBLE_DEVICES=0
model_path="model/yolov3/epoch_36.pth"
python tools/test.py ./outputs/yolov3/yolov3_d53_8xb8-ms-416-273e_coco.py  $model_path --out "./output_test/yolov3.pkl" --work-dir="./output_test"
