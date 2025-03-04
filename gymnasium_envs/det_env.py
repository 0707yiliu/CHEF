# MODEL: AprilTag markers detection and camera calibration and YOLO detection part
# AUTHOR: Yi Liu @AiRO
# UNIVERSITY: UGent-imec
# DEPARTMENT: Faculty of Engineering and Architecture
# Control Engineering / Automation Engineering

import cv2
import apriltag
import numpy as np
import matplotlib.pyplot as plt
import sys, os, math, time
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs
import json
from ultralytics import YOLO


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        pass

def dectshow(org_img, boxs):
    img = org_img.copy()
    print(len(boxs))
    for box in boxs:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(img, box[-1], (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('dec_img', img)

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results

def predict_and_detect(obj_list,
                       chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    '''
    names: {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
    '''
    obj_w_center = 0
    obj_h_center = 0
    for result in results:
        for box in result.boxes:
            if result.names[int(box.cls[0])] == obj_list:
                obj_w_center = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                obj_h_center = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
                cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                              (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
                cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                            (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), text_thickness)
    return img, results, [obj_w_center, obj_h_center]

class AprilTagDet_Depth:
    def __init__(self, rootid=9, objid=10,
                 yolo_det=False,
                 enable_recording=False, path=None, render=False) -> None:
        self.pipeline = rs.pipeline()  # define the pipeline
        self.config = rs.config()  # define the configuration
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)  # config the depth stream for detection
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # config the color stream
        self.profile = self.pipeline.start(self.config)
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        self.render = render

        self.FHD_fx = 914.676513671875 # 1280,720
        self.FHD_fy = 912.8101196289062
        self.FHD_cx = 645.1201171875
        self.FHD_cy = 372.2779541015625

        # self.FHD_fx = 1372.0147705078125 # 1920, 1080
        # self.FHD_fy = 1369.2152099609375
        # self.FHD_cx = 967.6801147460938
        # self.FHD_cy = 558.4169311523438

        self.enable_record = enable_recording
        self.recording_count = 0
        self.recording_path = path

        if self.enable_record is True:
            fps, w, h = 30, 1920, 1080
            mp4 = cv2.VideoWriter_fourcc(*'mp4v')
            self.wr = cv2.VideoWriter(self.recording_path, mp4, fps, (w, h), isColor=True)

        print("complete the RealSense initialization.")

        self.K = np.array([[self.FHD_fx, 0., self.FHD_cx],
                           [0., self.FHD_fy, self.FHD_cy],
                           [0., 0., 1.]])
        self.K1 = np.array([self.FHD_fx, self.FHD_fy, self.FHD_cx, self.FHD_cy])

        self.id_root = rootid
        self.id_object = objid
        self.tag_len = 8
        self.tag_outer_side = 0  # the box around apriltag
        obj_offset_x = 0  # to relative obj
        obj_offset_y = 0  # to relative obj
        obj_offset_z = 0  # to relative obj
        root_z_offset = 0  # to root of robot
        root_base_x = 0  # to root of robot
        root_base_y = 0  # to root of robot

        self.rootTobj = np.identity(4)
        self.rootTrootside = np.identity(4)
        self.rootsideTcam = np.identity(4)
        self.camTobjside = np.identity(4)
        self.objsideTobj = np.identity(4)

        self.robotTocam = np.identity(4)
        self.camTgraspobj = np.identity(4)
        self.camTgraspobj[1, 1] = 0
        self.camTgraspobj[2, 2] = 0
        self.camTgraspobj[1, 2] = 1
        self.camTgraspobj[2, 1] = -1

        self.rootTrootside[0, 3] = ((root_base_x / 2) - (self.tag_len / 2 + self.tag_outer_side))
        self.rootTrootside[1, 3] = (self.tag_len / 2 + self.tag_outer_side + root_base_y / 2)
        self.rootTrootside[2, 3] = root_z_offset
        self.objsideTobj[0, 3] = -obj_offset_x
        self.objsideTobj[1, 3] = -obj_offset_y
        self.objsideTobj[2, 3] = -obj_offset_z

        self.x = 0
        self.y = 0
        self.z = 0

        self.output = np.zeros(6)
        if yolo_det is True:
            self.yolo_v11_init()

    def yolo_v11_init(self):
        self.yolo_model = YOLO('yolo11m.pt')  #hard code for yolo setting

    def yolo_v11_det(self, model, color_image, obj_name):
        obj_list = ['apple', 'orange', 'banana'] # TODO: use for search, but we need the specified object (obj_name)
        _, _, obj_xy_coordinate = predict_and_detect(obj_name, model, color_image, conf=0.5)
        return obj_xy_coordinate

    def get_aligned_images(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        # intr = color_frame.profile.as_video_stream_profile().intrinsics
        # camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
        #                      'ppx': intr.ppx, 'ppy': intr.ppy,
        #                      'height': intr.height, 'width': intr.width,
        #                      'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
        #                      }
        color_image = np.asanyarray(color_frame.get_data())

        # depth image
        depth_frame = aligned_frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())

        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics  #

        return depth_intrin, color_image, depth_image, depth_frame

    def get_3d_camera_coordinate(self, _pixel, aligned_depth_frame, depth_intrin):
        x = _pixel[0]
        y = _pixel[1]
        dis = aligned_depth_frame.get_distance(x, y)  #
        # print ('depth: ',dis)   # unit:m
        camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, _pixel, dis)
        # print ('camera_coordinate: ',camera_coordinate)
        return dis, np.array(camera_coordinate)

    def get_coordinates(self):
        depth_intrin, img, depth_image, depth_frame = self.get_aligned_images()
        rootTotag = self.robot2tag(img)
        rootToobj = self.robot2obj(6, 'apple',
                                   depth_intrin, img, depth_image, depth_frame)
        print(rootToobj)

    def robot2obj(self, robot_tag_id, obj_name,
                  depth_intrin, img, depth_image, depth_frame):
        """
        detect the object position xyz in realsense frame, one object or one class of object
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        at_detactor = apriltag.Detector(apriltag.DetectorOptions(families="tag36h11"))
        tags = at_detactor.detect(gray)
        for tag in tags:
            M, e1, e2 = at_detactor.detection_pose(tag, self.K1)
            _t = M[:3, 3]
            t = self.tag_len * _t
            M[:3, 3] = t
            if tag.tag_id == robot_tag_id:
                self.robotTocam = np.linalg.inv(M)
        xy_camera_coordinate = self.yolo_v11_det(self.yolo_model, img, obj_name)  # the xy position in color image
        _, obj_pos_in_camera = self.get_3d_camera_coordinate(xy_camera_coordinate, depth_frame, depth_intrin)
        self.camTgraspobj[:3, -1] = obj_pos_in_camera * 100
        robot2graspobj = np.matmul(self.robotTocam, self.camTgraspobj)
        robot2graspobj_x = robot2graspobj[0, -1] / 100
        robot2graspobj_y = -robot2graspobj[1, -1] / 100
        robot2graspobj_z = -robot2graspobj[2, -1] / 100
        return np.array([robot2graspobj_x, robot2graspobj_y, robot2graspobj_z])

    def robot2tag(self) -> np.ndarray:
        depth_intrin, img, depth_image, depth_frame = self.get_aligned_images()
        cv2.waitKey(1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        at_detactor = apriltag.Detector(apriltag.DetectorOptions(families="tag36h11"))
        tags = at_detactor.detect(gray)

        for tag in tags:
            if self.render is True:
                H = tag.homography
                num, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, self.K)
                r = R.from_matrix(Rs[3].T)
                for i in range(4):
                    cv2.circle(img, tuple(tag.corners[i].astype(int)), 4, (255, 0, 0), 2)

            M, e1, e2 = at_detactor.detection_pose(tag, self.K1)
            P = M[:3, :4]
            _t = M[:3, 3]
            t = self.tag_len * _t
            P = np.matmul(self.K, P)
            self.z = np.matmul(P, np.array([[0], [0], [-1], [1]]))
            # print(z)
            self.z = self.z / self.z[2]
            self.x = np.matmul(P, np.array([[1], [0], [0], [1]]))
            self.x = self.x / self.x[2]
            self.y = np.matmul(P, np.array([[0], [-1], [0], [1]]))
            self.y = self.y / self.y[2]

            if self.render is True:
                cv2.line(img, tuple(tag.center.astype(int)), tuple(np.squeeze(self.x[:2].T, axis=0).astype(int)), (0, 0, 255), 2)
                cv2.line(img, tuple(tag.center.astype(int)), tuple(np.squeeze(self.y[:2].T, axis=0).astype(int)), (0, 255, 0), 2)
                cv2.line(img, tuple(tag.center.astype(int)), tuple(np.squeeze(self.z[:2].T, axis=0).astype(int)), (255, 0, 0), 2)

            M[:3, 3] = t
            if tag.tag_id == self.id_root:
                self.rootsideTcam = np.linalg.inv(M)
            elif tag.tag_id == self.id_object:
                self.camTobjside = np.copy(M)
                # print(self.camTobjside[:3,-1]/100)
            rootsideTobjside = np.matmul(self.rootsideTcam, self.camTobjside)
            rootTobjside = np.matmul(self.rootTrootside, rootsideTobjside)
            self.rootTobj = np.matmul(rootTobjside, self.objsideTobj)
            # now we just output x, y, z (unit: meter)
            self.x = self.rootTobj[0, 3] / 100
            self.y = -self.rootTobj[1, 3] / 100
            self.z = -self.rootTobj[2, 3] / 100
            # print(self.rootTobj[:3, :3])
            # [[ 0.99989305 -0.00309032 -0.01429449]
            #  [ 0.00284012  0.999843   -0.01749017]
            #  [ 0.0143463   0.0174477   0.99974485]]
            r = R.from_matrix(self.rootTobj[:3, :3])
            rot = r.as_rotvec()
            self.output = np.hstack([np.array([self.x, self.y, self.z]), rot])
        # print(self.output)


        # for calculating robot to object

        robot2cam = self.camTobjside[:3, -1] / 100 # robot root xyz, unit: m
        # print(robot2cam)
        xy_camera_coordinate = self.yolo_v11_det(self.yolo_model, img, 'apple') # the xy position in color image
        _, obj_pos_in_camera = self.get_3d_camera_coordinate(xy_camera_coordinate, depth_frame, depth_intrin)
        # print(obj_pos_in_camera) # embed the xyz
        # print('robot root:', robot2cam)
        # print('robot rot:', self.camTobjside[:3, :3])
        # print(type(obj_pos_in_camera))
        self.camTgraspobj[:3, -1] = obj_pos_in_camera * 100
        robot2camreal = np.linalg.inv(self.camTobjside)
        # print(robot2camreal, self.camTobjside)
        robot2graspobj = np.matmul(robot2camreal, self.camTgraspobj)
        # print('obj:', robot2graspobj[:3, -1] / 100)
        robot2graspobj_x = robot2graspobj[0, -1] / 100
        robot2graspobj_y = -robot2graspobj[1, -1] / 100
        robot2graspobj_z = -robot2graspobj[2, -1] / 100
        print(np.array([robot2graspobj_x, robot2graspobj_y, robot2graspobj_z]))

        cv2.imshow("camera-image", img)
        if self.enable_record is True:
            self.wr.write(img)
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('RealSense', colorizer_depth)
            cv2.imshow('RealSense', img)
            print('detTag recording...')
        # if self.render is True:
        #     cv2.imshow("camera-image", img)
        #     if cv2.waitKey(1) & 0xFF == ord("j"):
        #         i += 1
        #         n = str(i)
        #         filename = str("./image" + n + ".jpg")
        #         cv2.imwrite(filename, img)
        return self.output

# for testing
test = AprilTagDet_Depth(rootid=5, objid=6, yolo_det=True, render=True)
while True:
    # test.get_coordinates()
    test.robot2tag()