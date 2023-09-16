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
from src.lib.net.post_processing import nms
from src.lib.net.post_processing import pose_outputs as poseOut
from src.lib.net.post_processing import eval3d
from src.lib.net.post_processing.eval3d import measure_3d_iou, EvalMetrics, measure_ADD
from src.lib.net.post_processing.segmentation_outputs import draw_segmentation_mask_gt
from src.lib.net.post_processing.epnp import optimize_for_9D

ckpt_path = '/home/siheng/dha/views=2-epoch=112-step=411998.ckpt'
model_file = '/home/siheng/dha/MVTrans/model/multiview_net.py'
model_name = 'res_fpn'
hparam_file= '/home/siheng/dha/MVTrans/model/config/net_config_blender_multiview_2view_eval.txt'
model_path = (model_file)

class Detection:
    def __init__(self, camera_T_object=None, scale_matrix=None, box=None, obj_CAD=None):
      self.camera_T_object=camera_T_object #np.ndarray
      self.scale_matrix= scale_matrix # np.ndarray
      self.size_label="small"
      self.ignore = False
      self.box = box
      self.obj_CAD=0

def get_obj_pose_and_bbox(heatmap_output, vertex_output, z_centroid_output, cov_matrices, camera_model):
    peaks = poseOut.extract_peaks_from_centroid(np.copy(heatmap_output), max_peaks=np.inf)
    bboxes_ext = poseOut.extract_vertices_from_peaks(np.copy(peaks), np.copy(vertex_output), np.copy(heatmap_output))  # Shape: List(np.array([8,2])) --> y,x order
    z_centroids = poseOut.extract_z_centroid_from_peaks(np.copy(peaks), np.copy(z_centroid_output))
    cov_matrices = poseOut.extract_cov_matrices_from_peaks(np.copy(peaks), np.copy(cov_matrices))
    poses = []
    for bbox_ext, z_centroid, cov_matrix, peak in zip(bboxes_ext, z_centroids, cov_matrices, peaks):
        bbox_ext_flipped = bbox_ext[:, ::-1] # Switch from yx to xy
        # Solve for pose up to a scale factor
        error, camera_T_object, scale_matrix = optimize_for_9D(bbox_ext_flipped.T, camera_model, solve_for_transforms=True)
        abs_camera_T_object, abs_object_scale = epnp.find_absolute_scale(
            -1.0 * z_centroid, camera_T_object, scale_matrix
        )
        poses.append(transform.Pose(camera_T_object=abs_camera_T_object, scale_matrix=abs_object_scale))
    return poses, bboxes_ext

def get_obj_name(scene):
    return scene[0].split("/")[-3]

def prune_state_dict(state_dict):
  for key in list(state_dict.keys()):
    state_dict[key[6:]] = state_dict.pop(key)
  return state_dict

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

# model.cuda()
model.eval()

# load data from dataloader
val_ds = datapoint.make_dataset(hparams.val_path, dataset = 'blender', multiview = True, num_multiview = hparams.num_multiview, num_samples = hparams.num_samples)
data_loader = common.get_loader(hparams, "val", datapoint_dataset=val_ds)
data = next(iter(data_loader))
step = 1
obj_name = None
step_model = 0

# inference
if hparams.network_type == 'simnet':
    image, seg_target, depth_target, pose_targets, box_targets, keypoint_targets, _, scene_name = data
    seg_output, depth_output, small_depth_output, pose_outputs, box_outputs, keypoint_outputs = model.forward(
      image, step = step_model
  )
    step_model +=1
elif hparams.network_type == 'multiview':
    image, camera_poses, camera_intrinsic, seg_target, depth_target, pose_targets, _, scene_name = data
    camera_intrinsic=[item for item in camera_intrinsic]

    assert image.shape[1] == camera_poses.shape[1], f'dimension mismatch: num of imgs {image.shape} not equal to num of camera poses {camera_poses.shape}'

    seg_output, depth_output, small_depth_output, pose_outputs, box_outputs, keypoint_outputs = model.forward(
      imgs = image, cam_poses = camera_poses, cam_intr = camera_intrinsic, mode = 'val'
  )
else:
    raise ValueError(f'Network type not supported: {hparams.network_type}')

camera_model = camera.BlenderCamera()
with torch.no_grad():
    left_image_np = extract_left_numpy_img(image[0], mode = 'multiview')
    depth_vis = depth_output.get_visualization_img(np.copy(left_image_np))

    depth_target[0].depth_pred=np.expand_dims(depth_target[0].depth_pred, axis=0)
    depth_target[0].convert_to_torch_from_numpy()
    gt_depth_vis = depth_target[0].get_visualization_img(np.copy(left_image_np))

    seg_vis = seg_output.get_visualization_img(np.copy(left_image_np))
    seg_target[0].convert_to_numpy_from_torch()
    gt_seg_vis = draw_segmentation_mask_gt(np.copy(left_image_np), seg_target[0].seg_pred)

    c_img = cv2.cvtColor(np.array(left_image_np), cv2.COLOR_BGR2RGB)
    pose_vis = pose_outputs.get_visualization_img(np.copy(left_image_np), camera_model=camera_model)
    gt_pose_vis = pose_targets[0].get_visualization_img_gt(np.copy(left_image_np), camera_model=camera_model)

    # plotting
    rows = 2
    columns = 3
    fig = plt.figure(figsize=(15, 15))

    fig.add_subplot(rows, columns, 1)
    plt.imshow(gt_seg_vis)
    plt.axis('off')
    plt.title("gt_seg map")

    fig.add_subplot(rows, columns, 2)
    plt.imshow(gt_depth_vis)
    plt.axis('off')
    plt.title("gt depth map")

    fig.add_subplot(rows, columns, 3)
    plt.imshow(gt_pose_vis.astype(int))
    plt.axis('off')
    plt.title("gt pose vis")

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
