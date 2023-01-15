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
from std_msgs.msg import String
import subprocess



class PCLProcessor:
  def __init__(self, subscribe=False):
    self.rs_helpers = RealsenseHelpers()
    self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model and image preprocessing
    self.model, self.preprocess = clip.load("ViT-L/14@336px", device=self.device, jit=False)
    # print(clip.available_models())
    if subscribe:
      rospy.init_node('segment_pcl', anonymous=True)
      # rospy.Subscriber("/camera/depth/color/points", PointCloud2, self.pcl_callback, queue_size=1)
      rospy.Subscriber("/find_objects", String, self.find_objects_cb, queue_size=1)
  
  def calc_box_area(self, box):
    return max(0, box[1][0] - box[0][0] + 1) * max(0, box[1][1] - box[0][1] + 1)
  
  def calc_int_area(self, boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0][0], boxB[0][0])
    yA = max(boxA[0][1], boxB[0][1])
    xB = min(boxA[1][0], boxB[1][0])
    yB = min(boxA[1][1], boxB[1][1])
    # compute the area of intersection rectangle
    return self.calc_box_area([(xA, yA), (xB, yB)])
  
  def find_objects_cb(self, msg):
    """Find objects in the scene and publish their centroids
    centroids are published as a string of floats in the form:
    "x1 y1 z1 x2 y2 z2 x3 y3 z3"
    """
    detected_objects, object_points_wrt_aruco, new_object_centroids = self.find_object(msg.data)
    string_centroids = ""
    for centroid in new_object_centroids:
      string_centroids += f"{centroid[0]} {centroid[1]} {centroid[2]} "
  
  def get_pcd(self):
    msg = rospy.wait_for_message("/camera/depth/color/points", PointCloud2)
      # msg2 = rospy.wait_for_message("/astra/depth_registered/points", PointCloud2)
    rospy.loginfo("Received PointCloud2 message")
    pcd = orh.rospc_to_o3dpc(msg) 
    return pcd
  
  def display_inlier_outlier(self, cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])
  
  def transform_pcl_points(self, source_frame='camera_color_optical_frame', target_frame='aruco_base', points=None, pcd=None):
    if pcd is not None:
      np_points = np.asarray(pcd.points)
    elif points is not None:
      np_points = points
    else:
      raise ValueError("Must provide either points or pcd")
    dist_mat = np_points.T
    dist_mat = transform_dist_mat(dist_mat, source_frame, target_frame)
    np_points = dist_mat.T
    if points is not None:
      return np_points
    else:
      pcd.points = o3d.utility.Vector3dVector(np_points)
      return pcd
  
  def constrain_and_segment_plane(self, pcd, limits, visualize=False):
    np_points = np.asarray(pcd.points)
    np_points = self.transform_pcl_points(points=np_points)
    x_cond = np.logical_and(np_points[:, 0] >= limits["x_min"], np_points[:, 0] <= limits["x_max"])
    y_cond = np.logical_and(np_points[:, 1] >= limits["y_min"], np_points[:, 1] <= limits["y_max"])
    z_cond = np.logical_and(np_points[:, 2] >= limits["z_min"], np_points[:, 2] <= limits["z_max"])
    filtered_np_points = np.where(np.logical_and(x_cond, np.logical_and(y_cond, z_cond)))
    print("Number of points in plane: ", filtered_np_points[0].shape[0])
    if filtered_np_points[0].shape[0] < 10:
      return None, None, None
    pcd.points = o3d.utility.Vector3dVector(np_points[filtered_np_points])
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.017, ransac_n=3, num_iterations=1000)
    # print(plane_model)
    plane_cloud = pcd.select_by_index(inliers)
    # plane_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16), fast_normal_computation=True)
    objects_cloud = pcd.select_by_index(inliers, invert=True)
    plane_cloud.paint_uniform_color([1, 0, 0])
    # outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
    # objects_cloud = pcd
    # objects_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16), fast_normal_computation=True)
    # objects_cloud.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 0]))
    if visualize:
      o3d.visualization.draw_geometries([objects_cloud])
    return objects_cloud, plane_cloud, pcd
  
  def segment_pcl(self, visualize=False, verbose=False, msg=None):
    pcd = self.get_pcd()
    # limits = {'x_min': -0.636, 'x_max': 0.283,
    #           'y_min': -2.0, 'y_max': 0.383,
    #           'z_min': 0.42, 'z_max': 1.019}
    limits = {'x_min': -0.636, 'x_max': 0.358,
              'y_min': -0.061, 'y_max': 0.292,
              'z_min': 0.471, 'z_max': 2.0}
    objects_cloud, plane_cloud, pcd = self.constrain_and_segment_plane(pcd, limits, visualize=visualize)
    labels = np.array(objects_cloud.cluster_dbscan(eps=0.007, min_points=25))
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
      # pixels = np.round(pixels).astype(np.uint16)
      pixels = self.rs_helpers.adjust_pixels_to_boundary(
      pixels, (image_np.shape[1], image_np.shape[0]))
      rh = 8
      rv_up = 25
      rv_down = 18
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
      execluded_object_names = []
      # execluded_object_names.extend([f"not a {n}" for n in object_names])
      # execluded_object_names.append("other")
      # execluded_object_names.append("unknown")
      # execluded_object_names.append("something else")
      execluded_object_names.append("object")
      new_object_names.extend(execluded_object_names)
      text_snippets = ["a photo of a {}".format(name) for name in new_object_names]
      # pre-process text
      text = clip.tokenize(text_snippets).to(self.device)
      
      # with torch.no_grad():
      #     text_features = model.encode_text(text)
      detected_objects = []
      new_object_centroids = []
      objects_roi_by_class = {i:[] for i in object_names}
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
          correct = False
          if obj_idx < len(object_names):
            if object_names[obj_idx] == "cooking pot with lid":
              if probs[0][obj_idx] > 0.85:
                correct = True
              elif probs[0][obj_idx+1] > 0.25:
                obj_idx += 1
                correct = True
          if (probs[0][obj_idx] > 0.85) or correct:
              name = new_object_names[obj_idx]
              if name in execluded_object_names:
                continue
              if verbose:
                print("Object {} is {}".format(i, name))
              detected_objects.append({'name':name, 'roi':objects_on_table_roi[i], 'idx':i})
              new_object_centroids.append(object_centroids_wrt_aruco[i])
              if name not in execluded_object_names:
                objects_roi_by_class[name].append((i, probs[0][obj_idx]))
      indices_to_remove = []
      for _, class_data in objects_roi_by_class.items():
          class_indices = [i[0] for i in class_data]
          class_probs = [i[1] for i in class_data]
          class_rois = [objects_on_table_roi[i] for i in class_indices]
          if len(class_rois) > 1:
              areas = list(map(self.calc_box_area, class_rois))
              for i in range(len(class_rois)):
                  for j in range(i+1, len(class_rois)):
                      inter_area = self.calc_int_area(class_rois[i], class_rois[j])
                      if inter_area/min(areas[i], areas[j]) >= 0.1:
                          indices_to_remove = [class_indices[i] if class_probs[i] < class_probs[j] else class_indices[j]]
                          
      copy_of_detected_objects = deepcopy(detected_objects)
      for detected_object in copy_of_detected_objects:
          if detected_object['idx'] in indices_to_remove:
              detected_objects.remove(detected_object)
      
      if visualize_result:
        for detected_object in copy_of_detected_objects:
            # print("Object {} is {}".format(detected_object['idx'], detected_object['name']))
            if detected_object is not None:
                c  = (0, 0, 255) if detected_object['idx'] in indices_to_remove else (0, 255, 0)
                cv2.rectangle(image_np, (detected_object['roi'][0][0], detected_object['roi'][0][1]), (detected_object['roi'][1][0], detected_object['roi'][1][1]), c, 2)
                cv2.putText(image_np, detected_object['name'], (detected_object['roi'][0][0], detected_object['roi'][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
                # image_np[object_pixels[detected_object['idx']][1], object_pixels[detected_object['idx']][0]] = [255, 0, 0]
        # if len(indices_to_remove) > 0:
            # cv2.imwrite("image_with_removed_objects.png", image_np)
        # if len(detected_objects) >= len(object_names)-1:
        #   all_names = [detected_object['name'] for detected_object in detected_objects]
        #   found = [False if name not in all_names else True for name in object_names]
        #   if found[0] or found[1] :
        #     found[0] = True
        #     found[1] = True
        #   if all(found):
        #     cv2.imwrite("image_with_detected_objects.png", image_np)
        cv2.imshow("image", image_np)
        cv2.waitKey(10)
      return detected_objects, object_points_wrt_aruco, new_object_centroids
  
  def get_object_location(self, n_trials=5, object_names=["cup", "bottle", "tea packet"], number=False):
    found = [False for name in object_names]
    detected_objects_dict = {}
    for i in range(n_trials):
        detected_objects, object_points_wrt_aruco, object_centroids_wrt_aruco = self.find_object(object_names=object_names)
        all_names = [detected_object['name'] for detected_object in detected_objects]
        found = [False if name not in all_names else True for name in object_names]
        if all(found):
            object_numbers = {}
            for detected_object, center in zip(detected_objects, object_centroids_wrt_aruco):
                name = detected_object['name']
                new_name = name
                if name not in object_numbers.keys():
                    object_numbers[name] = 0
                object_numbers[name] += 1
                if number:
                    new_name = name + str(object_numbers[name])
                # np_points = np.array([object_points_wrt_aruco[detected_object['idx']].T, center])
                # np_points = self.transform_pcl_points(points=np_points, source_frame="aruco_base", target_frame="base_link").T
                top_idx = np.argmax(object_points_wrt_aruco[detected_object['idx']][1, :], axis=0)
                bottom_idx = np.argmin(object_points_wrt_aruco[detected_object['idx']][1, :], axis=0)
                top = object_points_wrt_aruco[detected_object['idx']][:, top_idx].tolist()
                bottom = object_points_wrt_aruco[detected_object['idx']][:, bottom_idx].tolist()
                detected_objects_dict[new_name] = {'center':center, 'top':top, 'bottom':bottom}
            return detected_objects_dict
    return None


if __name__ == '__main__':
  pcl_processor = PCLProcessor(subscribe=False)
  rospy.init_node('segment_pcl', anonymous=True)
  while not rospy.is_shutdown():
    try:
      # obj_names = ["cup", "bottle", "tea packet", "other"]
      obj_names = ["tea-packet"]
      obj_names = ["Tea-Packet"]
      obj_names = ['cooking pot with lid', 'cooking pot without lid', 'carrot', 'tomato', 'cup', 'tea-packet', 'cucumber', 'bottle']
      obj_names = ['bowl']
      # obj_names = ['carrot']
      # obj_names = ["cup"]
      # obj_names = ["cup", "tea packet"]
      # not_object_names = [f"something that does not look like a {n}" for n in obj_names]
      # new_object_names = obj_names + not_object_names + ["other"]
      pcl_processor.find_object(object_names=obj_names, verbose=True, visualize_steps=False)
    except rospy.ROSInterruptException:
      print("Shutting down")
      cv2.destroyAllWindows()
      break 