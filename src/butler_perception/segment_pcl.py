#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2, Image
from std_msgs.msg import String
from open3d_ros_helper import open3d_ros_helper as orh
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image as PILImage
from perception.laptop_perception_helpers import RealsenseHelpers, transform_dist_mat
from robot_helpers.srv import LookupTransform, TransformPoses
import cv2
from ros_numpy import numpify
import clip
import torch
from copy import deepcopy


class PCLProcessor:
  def __init__(self):
    # rospy.Subscriber("/camera/depth/color/points", PointCloud2, self.pcl_callback, queue_size=1)
    self.rs_helpers = RealsenseHelpers()
    self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model and image preprocessing
    self.model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
  
  def segment_pcl(self, visualize=False, verbose=False):
    msg = rospy.wait_for_message("/camera/depth/color/points", PointCloud2)
    rospy.loginfo("Received PointCloud2 message")
    pcd = orh.rospc_to_o3dpc(msg) 
    np_points = np.asarray(pcd.points)
    dist_mat = np_points.T
    dist_mat = transform_dist_mat(dist_mat, 'camera_color_optical_frame', 'aruco_base')
    np_points = dist_mat.T
    limits = {'x_min': -0.6359999999999999, 'x_max': 0.2829999999999999, 'y_min': -2.0, 'y_max': 0.383, 'z_min': 0.41999999999999993, 'z_max': 1.0190000000000001}
    x_cond = np.logical_and(np_points[:, 0] >= limits["x_min"], np_points[:, 0] <= limits["x_max"])
    y_cond = np.logical_and(np_points[:, 1] >= limits["y_min"], np_points[:, 1] <= limits["y_max"])
    z_cond = np.logical_and(np_points[:, 2] >= limits["z_min"], np_points[:, 2] <= limits["z_max"])
    filtered_np_points = np.where(np.logical_and(x_cond, np.logical_and(y_cond, z_cond)))
    pcd.points = o3d.utility.Vector3dVector(np_points[filtered_np_points])
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.016, ransac_n=3, num_iterations=1000)
    # print(plane_model)
    plane_cloud = pcd.select_by_index(inliers)
    # plane_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16), fast_normal_computation=True)
    objects_cloud = pcd.select_by_index(inliers, invert=True)
    plane_cloud.paint_uniform_color([1, 0, 0])
    # outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
    # objects_cloud = pcd
    if visualize:
      o3d.visualization.draw_geometries([objects_cloud])
    labels = np.array(objects_cloud.cluster_dbscan(eps=0.01, min_points=10))
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label 
    if max_label > 0 else 1))
    colors[labels < 0] = 0
    objects_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    if visualize:
      o3d.visualization.draw_geometries([objects_cloud])
    msg = rospy.wait_for_message("/camera/color/image_raw", Image)
    image_np = numpify(msg)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    objects_boundaries = []
    object_pixels = []
    object_points_wrt_camera = []
    object_points_wrt_aruco = []
    object_centroids_wrt_aruco = []
    for i in range(max_label+1):
      label_indices = np.where(labels == i)[0]
      if len(label_indices) < 80:
        continue
      cluster = objects_cloud.select_by_index(label_indices)
      center = np.asarray(cluster.get_center())
      object_centroids_wrt_aruco.append(center)
      if verbose:
        print("center = ", center)
      # if visualize:
      #   new_cluster = deepcopy(cluster)
      #   new_cluster.points.extend([center])
      #   new_cluster.colors.extend([[0, 0, 1]])
      #   o3d.visualization.draw_geometries([new_cluster])
      points_wrt_aruco = np.asarray(cluster.points).T
      points = transform_dist_mat(points_wrt_aruco, 'aruco_base', 'camera_color_optical_frame')
      intrinsics = self.rs_helpers.get_intrinsics(self.rs_helpers.color_intrin_topic)
      extrinsics = self.rs_helpers.get_depth_to_color_extrinsics()
      extrinsics = None
      pixels = self.rs_helpers.calculate_pixels_from_points(points, intrinsics, cam_to_cam_extrinsics=extrinsics)
      # print(pixels)
      # pixels = np.round(pixels).astype(np.uint16)
      pixels = self.rs_helpers.adjust_pixels_to_boundary(
      pixels, (image_np.shape[1], image_np.shape[0]))
      rh = 0
      rv_up = 20
      rv_down = 10
      miny, minx = min(pixels[1])-rv_up, min(pixels[0])-rh
      maxy, maxx = max(pixels[1])+rv_down, max(pixels[0])+rh
      boundary_pixels = self.rs_helpers.adjust_pixels_to_boundary(
      np.array([[minx, maxx],[miny, maxy]]), (image_np.shape[1], image_np.shape[0]))
      miny, maxy = boundary_pixels[1]
      minx, maxx = boundary_pixels[0]
      if maxx - minx < 5 or maxy - miny < 5:
        continue
      objects_boundaries.append([(minx, miny),
                                 (maxx, maxy)])
      object_points_wrt_aruco.append(points_wrt_aruco)
      object_points_wrt_camera.append(points)
      object_pixels.append(pixels)
      if visualize:
        # print(objects_boundaries[-1])
        # print(image_np.shape)
        # print(minx, miny, maxx, maxy)
        cv2.imshow("image", image_np[miny:maxy, minx:maxx])
        val = cv2.waitKey(0) & 0xFF
    return objects_boundaries, image_np, object_pixels, object_points_wrt_camera, object_points_wrt_aruco, object_centroids_wrt_aruco
  
  def find_object(self, object_names, verbose=False, visualize_result=True, visualize_steps=False):
      objects_on_table_roi, image_np, object_pixels, object_points_wrt_camera,\
      object_points_wrt_aruco, object_centroids_wrt_aruco = self.segment_pcl(verbose=verbose, visualize=visualize_steps)
      # image = PILImage.fromarray(np.uint8(image_np)*255)
      objects_images = []
      for object_roi in objects_on_table_roi:
          obj_image = PILImage.fromarray(image_np[object_roi[0][1]:object_roi[1][1], object_roi[0][0]:object_roi[1][0]])
          # cv2.imshow("object_image", np.array(obj_image))
          # cv2.waitKey(0)
          objects_images.append(obj_image)
      new_object_names = deepcopy(object_names)
      new_object_names.append("unknown")
      text_snippets = ["a photo of a {}".format(name) for name in new_object_names]
      # pre-process text
      text = clip.tokenize(text_snippets).to(self.device)
      
      # with torch.no_grad():
      #     text_features = model.encode_text(text)
      detected_objects = []
      new_object_centroids = []
      for i, object_image in enumerate(objects_images):
          # pre-process image
          prepro_image = self.preprocess(object_image).unsqueeze(0).to(self.device)
          
          # with torch.no_grad():
          #     image_features = model.encode_image(prepro_image)
          
          with torch.no_grad():
              logits_per_image, logits_per_text = self.model(prepro_image, text)
              probs = logits_per_image.softmax(dim=-1).cpu().numpy()
          # print("Label probs:", ["{0:.10f}".format(i) for i in probs[0]])
          obj_idx = np.argmax(probs[0])
          if verbose:
            print("object_index =", obj_idx)
            print("probabilities = ", probs[0])
          if (probs[0][obj_idx] > 0.7):
              if verbose:
               print("Object {} is {}".format(i, new_object_names[obj_idx]))
              detected_objects.append({'name':new_object_names[obj_idx], 'roi':objects_on_table_roi[i], 'idx':i})
              new_object_centroids.append(object_centroids_wrt_aruco[i])
      
      if visualize_result:
        for detected_object in detected_objects:
            # print("Object {} is {}".format(detected_object['idx'], detected_object['name']))
            if detected_object is not None:
                cv2.rectangle(image_np, (detected_object['roi'][0][0], detected_object['roi'][0][1]), (detected_object['roi'][1][0], detected_object['roi'][1][1]), (0, 255, 0), 2)
                cv2.putText(image_np, detected_object['name'], (detected_object['roi'][0][0], detected_object['roi'][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # image_np[object_pixels[detected_object['idx']][1], object_pixels[detected_object['idx']][0]] = [255, 0, 0]
        cv2.imshow("image", image_np)
        cv2.waitKey(10)
      return detected_objects, object_points_wrt_aruco, new_object_centroids


if __name__ == '__main__':
  pcl_processor = PCLProcessor()
  rospy.init_node('segment_pcl', anonymous=True)
  while not rospy.is_shutdown():
    try:
      pcl_processor.find_object(object_names=["cup", "bottle", "tea packet", "other"], verbose=True, visualize_steps=False)
    except rospy.ROSInterruptException:
      print("Shutting down")
      cv2.destroyAllWindows()
      break