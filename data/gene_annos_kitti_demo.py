if __name__=='__main__':
    import os
    import os.path as osp
    import numpy as np
    import cv2
    import json

    code_root = '/data2/liuzhi/3DGS_code/Metric3D/'
    scene = "t100pro_in_talandB1"
    # data_root = osp.join(code_root, '../../Dataset/3DGS_Dataset/input/gm_Museum')
    data_root = osp.join(code_root, '../../remote_data/dataset_reality/test/' + scene)
    split_root = code_root

    files = []
    rgb_root = osp.join(data_root, 'images')
    depth_root = osp.join(data_root, 'depth')
    # 有子文件夹的全景相机数据集
    # rgb_root：/home/liuzhi/Disk_v100/3DGS_code/Metric3D/data/gm_Museum/rgb
    # for rgb_subfolder in os.listdir(rgb_root):
    #     for rgb_file in os.listdir(rgb_root + '/' + rgb_subfolder):
    #         rgb_path = osp.join(rgb_root, rgb_subfolder, rgb_file).split(split_root)[-1]
    #         # depth_path = rgb_path.replace('/rgb/', '/depth/')
    #         cam_in = [540.0000, 540.0000, 540.0000, 360.0000]
    #         depth_scale = 1.
    #
    #         meta_data = {}
    #         meta_data['cam_in'] = cam_in
    #         meta_data['rgb'] = rgb_path
    #         meta_data['depth'] = None
    #         meta_data['depth_scale'] = None
    #         files.append(meta_data)
    # files_dict = dict(files=files)

    # 普通数据集
    for rgb_file in os.listdir(rgb_root):
        rgb_path = osp.join(rgb_root, rgb_file).split(split_root)[-1]
        # depth_path = rgb_path.replace('/rgb/', '/depth/')
        cam_in = [480, 480, 480, 360]
        # cam_in = [, 556.281585, 480, 270.00]
        depth_scale = 1.

        meta_data = {}
        meta_data['cam_in'] = cam_in
        meta_data['rgb'] = rgb_path
        meta_data['depth'] = None
        meta_data['depth_scale'] = None
        files.append(meta_data)
    files_dict = dict(files=files)

    with open(osp.join(code_root, '../../remote_data/dataset_reality/test/' + scene + '/test_annotations.json'), 'w') as f:
        json.dump(files_dict, f)
        