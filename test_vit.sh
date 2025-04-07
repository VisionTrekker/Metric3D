export CUDA_VISIBLE_DEVICES=1
# Retail_Street, CBD_building_02
scenes=("t100pro_in_talandB1")

for scene in "${scenes[@]}"; do
  echo "estimate normal on: $scene"

  python mono/tools/test_scale_cano.py \
      'mono/configs/HourglassDecoder/vit.raft5.large.py' \
      --load-from ./weight/metric_depth_vit_large_800k.pth \
      --test_data_path ../../remote_data/dataset_reality/test/$scene/test_annotations.json \
      --show-dir ../../remote_data/dataset_reality/test/$scene/normal \
      --launcher None
done