import os
import argparse
import numpy as np
import json
from read_binary import read_cameras_binary, read_cameras_text

# 估计绝对深度
parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
parser.add_argument('--indir', required=True, type=str)
parser.add_argument('--outdir', required=False, type=str)
parser.add_argument('--load-from', dest='load_from', required=False, type=str, default='./weight/metric_depth_vit_large_800k.pth')
args = parser.parse_args()

# 1. 生成json文件
data_root = args.indir
rgb_root = os.path.join(data_root, 'images')
depth_root = os.path.join(data_root, 'depth')

try:
    cam_intrinsics = read_cameras_binary(os.path.join(data_root, 'sparse/0', "cameras.bin"))
except:
    cam_intrinsics = read_cameras_text(os.path.join(data_root, 'sparse/0', "cameras.txt"))


if cam_intrinsics[1].model == "SIMPLE_PINHOLE":
    fx = cam_intrinsics[1].params[0]
    fy = fx
    cx = cam_intrinsics[1].params[1]
    cy = cam_intrinsics[1].params[2]
elif cam_intrinsics[1].model == "PINHOLE":
    fx = cam_intrinsics[1].params[0]
    fy = cam_intrinsics[1].params[1]
    cx = cam_intrinsics[1].params[2]
    cy = cam_intrinsics[1].params[3]
intr = [fx, fy, cx, cy]


files = []
subfolders = os.listdir(rgb_root)
if len(subfolders)==4 and all(subfolder in ['00','01','02','03'] for subfolder in subfolders):
    # 分割为四个视角的全景相机数据
    for rgb_subfolder in subfolders:
        for rgb_file in os.listdir(rgb_root + '/' + rgb_subfolder):
            rgb_path = os.path.join(rgb_root, rgb_subfolder, rgb_file)
            # depth_path = rgb_path.replace('/rgb/', '/depth/')

            if not os.path.exists(rgb_path):
                continue

            cam_in = intr
            depth_scale = 1.

            meta_data = {}
            meta_data['cam_in'] = cam_in
            meta_data['rgb'] = rgb_path
            meta_data['depth'] = None
            meta_data['depth_scale'] = None
            files.append(meta_data)
else:
    for rgb_file in subfolders:
        rgb_path = os.path.join(rgb_root, rgb_file)
        # depth_path = rgb_path.replace('/rgb/', '/depth/')

        if not os.path.exists(rgb_path):
            continue

        cam_in = intr
        depth_scale = 1.

        meta_data = {}
        meta_data['cam_in'] = cam_in
        meta_data['rgb'] = rgb_path
        meta_data['depth'] = None
        meta_data['depth_scale'] = None
        files.append(meta_data)


files_dict = dict(files=files)
with open(os.path.join(data_root, 'test_annotations.json'), 'w') as f:
    json.dump(files_dict, f)


# 2. 估计
cmd = f"python mono/tools/test_scale_cano.py \
      'mono/configs/HourglassDecoder/vit.raft5.large.py' \
      --load-from {args.load_from} \
      --test_data_path {args.indir+'/test_annotations.json'} \
      --show-dir {args.indir+'/normal'} \
      --launcher None"
os.system(cmd)