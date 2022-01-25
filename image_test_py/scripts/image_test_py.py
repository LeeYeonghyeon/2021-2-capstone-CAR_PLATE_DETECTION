#!/usr/bin/env python

import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rospy

if __name__=='__main__':
    try:
        rospy.init_node('image_test_py_node', anonymous=True)
        pub = rospy.Publisher('/usb_cam/image_raw', Image, queue_size=1)
        
        cap = cv2.imread('/root/catkin_ws/src/image_test_py/scripts/1.jpg')
        bridge = CvBridge()
        ii=0
        while(True):
            frame = cap
            ii+=1
       
            pub.publish(bridge.cv2_to_imgmsg(frame, encoding = 'bgr8'))
            cv2.imwrite('/root/catkin_ws/src/image_test_py/scripts/target.jpg', frame)
            if ii > 1000:
                break
      
    except rospy.ROSInterruptException:
        pass
