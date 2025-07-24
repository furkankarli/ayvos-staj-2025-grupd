# Model Training Configuration
# The following command is used to train the model with the specified parameters.
python -m open_clip_train.main \
  --train-data "train.csv" \
  --val-data "val.csv" \
  --dataset-type "csv" \
  --csv-separator "," \
  --csv-img-key "image_path" \
  --csv-caption-key "model_name" \
  --model "ViT-L-14-quickgelu" \
  --pretrained "openai" \
  --batch-size 32 \
  --lr 5e-4 \
  --wd 0.01 \
  --epochs 100 \
  --warmup 500 \
  --workers 4 \
  --precision amp \
  --report-to "tensorboard" \
  --save-frequency 5 \
  --aug-cfg scale="0.9,1.1" color_jitter=0.3 \
  --name "cfv-vit-l14-improved3"