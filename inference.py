import numpy as np
import cv2
import torch
import os
import argparse
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader
from matplotlib import pyplot as plt
from PIL import Image
from importlib.machinery import SourceFileLoader

from src.lib import datapoint, camera, transform
from src.lib.net import common
from src.lib.net.init.default_init import default_init
from src.lib.net.dataset import extract_left_numpy_img
from src.lib.net.post_processing import epnp
from src.lib.net.post_processing import pose_outputs as poseOut
from src.lib.net.post_processing.segmentation_outputs import draw_segmentation_mask_gt
from src.lib.net.post_processing.epnp import optimize_for_9D
import json
import imageio
import glob

ckpt_path = '/home/siheng/dha/views=5-epoch=100-step=734169.ckpt'
model_file = '/home/siheng/dha/MVTrans/model/multiview_net.py'
hparam_file= '/home/siheng/dha/MVTrans/model/config/net_config_blender_multiview_2view_eval.txt'
model_name = 'res_fpn'
model_path = (model_file)

class Detection:
        def __init__(self, camera_T_object=None, scale_matrix=None, box=None, obj_CAD=None):
            self.camera_T_object=camera_T_object #np.ndarray
            self.scale_matrix= scale_matrix # np.ndarray
            self.size_label="small"
            self.ignore = False
            self.box = box
            self.obj_CAD=0

def prune_state_dict(state_dict):
    for key in list(state_dict.keys()):
        state_dict[key[6:]] = state_dict.pop(key)
    return state_dict


# load data from dataloader
# val_ds = datapoint.make_dataset(hparams.val_path, dataset = 'blender', multiview = True, num_multiview = hparams.num_multiview, num_samples = hparams.num_samples)
# data_loader = common.get_loader(hparams, "val", datapoint_dataset=val_ds)
# data = next(iter(data_loader))

def read_and_preprocess_image(img_path, blend_a=True):
        img = imageio.imread(img_path).astype(np.float32) / 255.0

        # Alpha blending, if necessary
        if img.shape[2] == 4:
                if blend_a:
                        img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])
                else:
                        img = img[..., :3] * img[..., -1:]

        return img

# GLOB all the files in validation and generate a depth map for EACH of the file.
# For each file, you have to construct the CAMERA_POSES tensor from reading a json.
# returns a list of filenames, 5 at a time.
# [[r_01, r_02, r_03, r_04, r_05], [r_02, r_03, r04, r_05, r_06] ... until you get to [r_0n] as first element.
# SUPER HACKY. FIX WHEN YOU HAVE TIME
def get_fn_from_validation_folder(folder):
    images = glob.glob(os.path.join(folder, 'r_*_0.png'))
    all_image_sets = []
    for i in range(0, len(images)-5):
        to_append = images[i:i+5]
        assert(len(to_append) == 5)
        all_image_sets.append(to_append)
    all_image_sets.append([images[len(images)-5], images[len(images)-4], images[len(images)-3], images[len(images)-2], images[len(images)-1]])
    all_image_sets.append([images[len(images)-4], images[len(images)-3], images[len(images)-2], images[len(images)-1], images[0]])
    all_image_sets.append([images[len(images)-3], images[len(images)-2], images[len(images)-1], images[0], images[1]])
    all_image_sets.append([images[len(images)-2], images[len(images)-1], images[0], images[1], images[2]])
    all_image_sets.append([images[len(images)-1], images[0], images[1], images[2], images[3]])

    assert(len(all_image_sets) == len(images))
    return all_image_sets

def prep_image_set_and_poses(one_image_set, json_filename):
    with open(json_filename, 'r') as file:
        data = file.read()
        json_data = json.loads(data)

    all_images = []
    all_transforms = []
    for image in one_image_set:
        for frame in json_data.get("frames", []):
                if frame.get("file_path") == os.path.basename(image[:-6]):
                    transform = np.array(frame.get("transform_matrix"))
                    break
                    
        img = read_and_preprocess_image(image)
        img = np.transpose(img, (2,0,1))
        all_images.append(img)
        all_transforms.append(transform)
    
    transforms = np.stack(all_transforms, axis=0)
    transforms = torch.from_numpy(transforms)
    transforms = transforms.unsqueeze(0)
    stacked_images = np.stack(all_images, axis=0)
    image = torch.from_numpy(stacked_images)
    image = image.unsqueeze(0)
    assert(image.shape == (1, 5,3,800,800))
    return image, transforms

def read_intrinsics(file):
    with open(file, 'r') as file:
        data = file.read()
        json_data = json.loads(data)
    return json_data['fl_x'], json_data['fl_y'], json_data['cx'], json_data['cy']

def construct_intrinsics(f_x, f_y, c_x, c_y):
    camera_intrinsics = [
            torch.tensor(
                    [[[f_x,   0.0000, c_x],
                    [  0.0000, f_y, c_y],
                    [  0.0000,   0.0000,   1.0000]]]
            )
    ]
    return camera_intrinsics

def main(folder_path, json_fp):
    model_path = (model_file)

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    common.add_train_args(parser)
    hparams = parser.parse_args(['@config/net_config_blender_multiview_2view_eval.txt'])

    print('Using model class from:', model_path)
    net_module = SourceFileLoader(model_name, str(model_path)).load_module()
    net_attr = getattr(net_module, model_name)
    model = net_attr(hparams)
    model.apply(default_init)

    print('Restoring from checkpoint:', ckpt_path)
    state_dict = torch.load(ckpt_path)['state_dict']
    state_dict = prune_state_dict(state_dict)
    model.load_state_dict(state_dict)

    model.cuda()
    model.eval()
    torch.cuda.empty_cache()

    f_x, f_y, c_x, c_y = read_intrinsics(json_fp)
    camera_intrinsics = construct_intrinsics(f_x, f_y, c_x, c_y)
    l_images = get_fn_from_validation_folder(folder_path)
    for image_set in l_images:     
        image, pose = prep_image_set_and_poses(image_set, json_fp)
        seg_output, depth_output, small_depth_output, pose_outputs, box_outputs, keypoint_outputs = model.forward(
            imgs = image.to("cuda"), cam_poses = pose.to("cuda"), cam_intr = camera_intrinsics, mode = 'val'
        )
        with torch.no_grad():
            depth_output.convert_to_numpy_from_torch()
            # save depth_output.depth_pred to numpy array
    else:
            raise ValueError(f'Network type not supported: {hparams.network_type}')

camera_model = camera.BlenderCamera()
with torch.no_grad():

    left_image_np = extract_left_numpy_img(image[0], mode = 'multiview')
        
    depth_vis = depth_output.get_visualization_img(np.copy(left_image_np))

    # depth_target[0].depth_pred=np.expand_dims(depth_target[0].depth_pred, axis=0)
    # depth_target[0].convert_to_torch_from_numpy()
    # gt_depth_vis = depth_target[0].get_visualization_img(np.copy(left_image_np))

    seg_vis = seg_output.get_visualization_img(np.copy(left_image_np))
    # seg_target[0].convert_to_numpy_from_torch()
    # gt_seg_vis = draw_segmentation_mask_gt(np.copy(left_image_np), seg_target[0].seg_pred)

    # c_img = cv2.cvtColor(np.array(left_image_np), cv2.COLOR_BGR2RGB)
    pose_vis = pose_outputs.get_visualization_img(np.copy(left_image_np), camera_model=camera_model)
    # gt_pose_vis = pose_targets[0].get_visualization_img_gt(np.copy(left_image_np), camera_model=camera_model)

    # plotting
    rows = 2
    columns = 3
    fig = plt.figure(figsize=(15, 15))

    # fig.add_subplot(rows, columns, 1)
    # plt.imshow(gt_seg_vis)
    # plt.axis('off')
    # plt.title("gt_seg map")

    # fig.add_subplot(rows, columns, 2)
    # plt.imshow(gt_depth_vis)
    # plt.axis('off')
    # plt.title("gt depth map")

    # fig.add_subplot(rows, columns, 3)
    # plt.imshow(gt_pose_vis.astype(int))
    # plt.axis('off')
    # plt.title("gt pose vis")

    fig.add_subplot(rows, columns, 4)
    plt.imshow(seg_vis)
    plt.axis('off')
    plt.title("seg map")

    fig.add_subplot(rows, columns, 5)
    plt.imshow(depth_vis)
    plt.axis('off')
    plt.title("depth map")

    fig.add_subplot(rows, columns, 6)
    plt.imshow(pose_vis.astype(int))
    plt.axis('off')
    plt.title("pose vis")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('folder_path')
    parser.add_argument('json_path')
    args = parser.parse_args()
    main(args.folder_path, args.json_path)
    
