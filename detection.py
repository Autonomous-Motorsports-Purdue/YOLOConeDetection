#!/usr/bin/env python
import struct
import colorsys
import rospy

from pathlib import Path
from models import *
from utils.datasets import *
from utils.utils import *

from sensor_msgs.msg import Image
from vision_msgs.msg import BoundingBox2D
from geometry_msgs.msg import Pose2D

from cv_bridge import CvBridge
bridge = CvBridge()

class Detection:
    def __init__(self, weight_path, cfg_path='cfg/yolov3.cfg'):
        self.img_size=416,
        self.conf_thres=0.3,
        self.nms_thres=0.45,
        
        self.device = torch_utils.select_device()
        
        # Get classes and colors
        # self.classes = load_classes(parse_data_cfg('cfg/coco.data')['names'])
        
        # Initialize model
        self.model = Darknet(cfg_path, self.img_size)
        
        # Load weights
        assert weight_path.endswith('.pt')
        self.model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
        
        self.model.to(self.device).eval()
        

    def image_callback(self, pub, pub_debug, image):
        
        with torch.no_grad():
            # Image formatting
            cv_image = bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
            np_image = np.asarray(cv_image)
            img = torch.from_numpy(np_image).unsqueeze(0).to(device)
            
            pred = model(img)
            pred = pred[pred[:, :, 4] > conf_thres]  # remove boxes < threshold

            if len(pred) == 0:
                return
                
            # Run NMS on predictions
            detections = non_max_suppression(pred.unsqueeze(0), self.conf_thres, self.nms_thres)[0]

            # Rescale boxes from 416 to true image size
            scale_coords(img_size, detections[:, :4], im0.shape).round()
            
            # Draw bounding boxes and labels of detections
            for x1, y1, x2, y2, conf, cls_conf, cls in detections:
                # Add bbox to the image
                bounding_box = new BoundingBox2D(
                    Pose2D((x1 + x2) / 2, (y1 + y2) / 2, 0)),
                    x2 - x1, y2 - y1
                ))
                
                label = plot_one_box([x1, y1, x2, y2], np_labels)
                pub_debug.publish(self.bridge.cv2_to_imgmsg(np_labels, "bgr8"))
                
                # print(label, end=', ')

                pub.publish(bounding_box)


if __name__ == "__main__":
    # TODO make this into action server
    try:
        rospy.init_node("cone_finder")
        pub = rospy.Publisher("yolo/bounding_boxes", BoundingBox2D)
        pub_debug = rospy.Publisher("yolo/bounding_boxes_debug", Image)
        
        detector = Detection("weights/best.pb", 'cfg/yolov3.cfg') 
        
        # Bind pub variable to callback
        image_callback_with_pub = lambda image: detector.image_callback(pub, pub_debug, image)
        
        rospy.Subscriber("/yolo/images", Image, image_callback_with_pub)
    except rospy.ROSInterruptException:
        pass
