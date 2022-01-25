#!/usr/bin/env python3
import argparse
from time import time
import sys
import numpy as np
import cv2
import tensorflow as tf
import math
import rospy
from sensor_msgs.msg import Image
from tools.model import LPRNet
from tools.loader import resize_and_normailze

classnames = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
              "가", "나", "다", "라", "마", "거", "너", "더", "러",
              "머", "버", "서", "어", "저", "고", "노", "도", "로",
              "모", "보", "소", "오", "조", "구", "누", "두", "루",
              "무", "부", "수", "우", "주", "허", "하", "호"
              ]

def callback(data):
    if(data):
        global fcount
        fcount += 1

#        if data.encoding != "bgr8":
#            rospy.logerr("test err")
        dtype = np.dtype("uint8")
        dtype = dtype.newbyteorder('>' if data.is_bigendian else '<')
        img = np.ndarray(shape=(data.height, data.width, 3), dtype=dtype, buffer=data.data)
        cv2.imwrite("/root/catkin_ws/src/Korean-License-Plate-Recognition/result/" + str(video_number) + "_" + str(fcount) + "_" + "original" + ".jpg", img)
        if data.is_bigendian == (sys.byteorder == 'little'):
            img = img.byteswap().newbyteorder()
        x = np.expand_dims(resize_and_normailze(img), axis=0)
        result1 = str(net.predict(x, classnames))
        print("[Korean License]: Before Warping Result = " + result1 + "")
#        cv2.imwrite("/root/catkin_ws/src/Korean-License-Plate-Recognition/result/" + str(video_number) + "_" + str(fcount) + "_" + str(net.predict(x, classnames)) + "nowarpping" + ".jpg", img)
        RECT_MODE = 0
        h, w, ch = img.shape

        ##############
        # Remove light
        img_YUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)    
        y = img_YUV[:,:,0]    
         
        rows = y.shape[0]    
        cols = y.shape[1]
         
        imgLog = np.log1p(np.array(y, dtype='float') / 255) 
         
        M = 2*rows + 1
        N = 2*cols + 1
         
        sigma = 10
        (X, Y) = np.meshgrid(np.linspace(0, N-1, N), np.linspace(0, M-1, M)) 
        Xc = np.ceil(N/2) 
        Yc = np.ceil(M/2)
        gaussianNumerator = (X - Xc)**2 + (Y - Yc)**2 
         
        LPF = np.exp(-gaussianNumerator / (2*sigma*sigma))
        HPF = 1 - LPF
         
        LPF_shift = np.fft.ifftshift(LPF.copy())
        HPF_shift = np.fft.ifftshift(HPF.copy())
         
        img_FFT = np.fft.fft2(imgLog.copy(), (M, N))
        img_LF = np.real(np.fft.ifft2(img_FFT.copy() * LPF_shift, (M, N)))
        img_HF = np.real(np.fft.ifft2(img_FFT.copy() * HPF_shift, (M, N)))

        ### 각 LF, HF 성분에 scaling factor를 곱해주어 조명값과 반사값을 조절함
        gamma1 = 0.6
        gamma2 = 0.0
        img_adjusting = gamma1*img_LF[0:rows, 0:cols] + gamma2*img_HF[0:rows, 0:cols]
         
        img_exp = np.expm1(img_adjusting) # exp(x) + 1
        img_exp = (img_exp - np.min(img_exp)) / (np.max(img_exp) - np.min(img_exp)) 
        img_out = np.array(255*img_exp, dtype = 'uint8') 
         
        img_YUV[:,:,0] = img_out
        result = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        
        ##############
        # divide Area
        v_th = 160
        cy = h/2
        cx = w/2
        ct1 = 0
        ct2 = 0
        for i in range(int(cy*0.8), int(cy*1.2)): 
            for j in range(int(cx*0.7), int(cx*1.3)):
                ct1 += hsv[i][j][2]
                ct2 += 1
        ct1 = ct1 / ct2
        if ct1 < v_th:
            v_th = ct1
        gray_img = cv2.inRange(hsv, (0, 0, v_th),(255,70,255-(255-v_th)*0.0)) #170
                
        ##############
        # Noise Remove
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (13,13))
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))
        gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, k)
        gray_img = cv2.erode(gray_img,k2)
        bigs = 0
        contect = None

        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(gray_img, connectivity=4)

        for i in range(retval):
            if bigs < stats[i][4] and  gray_img[labels == i][0] > 0:
                bigs = stats[i][4]
                contect = i
        gray_img[:] = 0
        gray_img[labels == contect] = 255
        gray_img = cv2.GaussianBlur(gray_img, ksize=(15, 15), sigmaX=0)
        
        ##############
        # Fitting Area
        img = img.copy()
        for h_ in range(h):
            for w_ in range(w):
                if gray_img[h_,w_] == 0:
                    img[h_,w_] = 0
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        gray_img = cv2.dilate(gray_img,k2)
        gray_image = cv2.GaussianBlur(gray_img, ksize=(15, 15), sigmaX=0)
        gray_image = cv2.Canny(gray_image,100,200)

        ##############
        # Correct Rotation
        H_th = 130
        for _ in range(4):
            lines = cv2.HoughLines(gray_image, 3, np.pi/180, H_th)
            angleSum = 0
            try:
                for i in range(len(lines)):
                    for rho, theta in lines[i]:
                        a, b = np.cos(theta), np.sin(theta)
                        x0, y0 = a*rho, b*rho
                        x1, y1 = int(x0 + 1000*(-b)), int(y0 + 1000*(a))
                        x2, y2 = int(x0 - 1000*(-b)), int(y0 - 1000*(a))
                        angleSum += math.degrees(math.atan2(y2-y1, x2-x1))
                break
            except:
                H_th = 80 - 15*_
                continue
        angle = angleSum / len(lines)
#        print(angle)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        check_img = cv2.warpAffine(img, M, (w,h))
        gray_img = cv2.warpAffine(gray_img, M, (w,h))
        
#        for h_ in range(h):
#            for w_ in range(w):
#                if not (check_img[h_,w_][0] > v_th and check_img[h_,w_][1] > v_th and check_img[h_,w_][2] > v_th):
#                    check_img[h_,w_] = 0
#        
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
        gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, k)
        for h_ in range(h):
            for w_ in range(w):
                if gray_img[h_,w_] > 0:
                    gray_img[h_,w_] = 255

        ##############
        # Estimation
        keypoints= cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        def Estimation_func():
            peri_th = 0.03
            ref_th = peri_th
            pts1 = []
            deadline = False
            while peri_th < ref_th *3 and len(pts1) == 0 and deadline == False:
                big = 0
                for _ in range(len(keypoints[0])):
                    if(RECT_MODE):
                        x, y,  w_, h_ = cv2.boundingRect(keypoints[0][_])
                        if w_*h_ > big:
                            xx = x
                            yy = y
                            big = w_*h_
                            ww = w_
                            hh = h_
                        deadline = True
                    else:
                        ret = cv2.contourArea(keypoints[0][_])
                        peri = cv2.arcLength(keypoints[0][_],True)
                        approx = cv2.approxPolyDP(keypoints[0][_], peri * peri_th,True)
                        if big < ret and len(approx) ==4:
                            if peri_th > ref_th and ret > (h*w)*0.125:
                                big = ret
                                pts1 = approx
                if RECT_MODE == 0 and  len(pts1) == 0:
                    peri_th = peri_th * 1.7
            if(RECT_MODE):
                return [xx, yy, ww, hh]
            return pts1
        
        pts1 = Estimation_func()
        if len(pts1) == 0:
            RECT_MODE = 1
            pts1 = Estimation_func()

        if(RECT_MODE):
            xx, yy ,ww, hh = pts1[0:]
            pts1 = np.float32([[xx, yy],[xx,yy+hh],[xx+ww, yy+hh],[xx+ww,yy]])
        else:
            pts1 = np.reshape(np.float32(pts1), (4,2))
            cols_id, row_id = np.argsort(pts1[:,0]), np.argsort(pts1[:,1])
            ls = []
            for n in [-1,-2]:
                if 0 == ls.count(cols_id[n]):
                    ls.append(cols_id[n])
                else:
                    n3 = cols_id[n] # 3
                if 0 == ls.count(row_id[n]):
                    ls.append(row_id[n])
                else:
                    n3 = row_id[n] # 3
            del ls[np.where(ls == n3)[0][0]]
            if (pts1[ls[0],0] > pts1[ls[1],0]):
                n4 = ls[0]
                n2 = ls[1]
            else:
                n4 = ls[1]
                n2 = ls[0]
            ls.append(n3)
            for n1 in range(3):
                if 0 == ls.count(n1):
                    break
            n1_x, n1_y, n2_x, n2_y, n3_x, n3_y, n4_x, n4_y = pts1[n1,0], pts1[n1,1], pts1[n2,0], pts1[n2,1], pts1[n3,0], pts1[n3,1], pts1[n4,0], pts1[n4,1] 
            pts1 = np.float32([[n1_x, n1_y],[n2_x,n2_y],[n3_x,n3_y],[n4_x,n4_y]])

        ##############
        # Perspective Transformation
        pts2 = np.float32([[0, 0],[0,h-1],[w-1, h-1],[w-1,0]])
        mtrx = cv2.getPerspectiveTransform(pts1, pts2)
        h_ = int(w/3)
        check_img = cv2.resize(cv2.warpPerspective(check_img, mtrx, (w,h)), (w,h_),cv2.INTER_AREA)
         
        ##############
        # Padding
        h=h_
        pad = 5
        inner_pad = 0
        pad_h = h+pad*2
        pad_w = w+pad*2
        check_img_ = np.ones((pad_h, pad_w, 3),dtype=np.uint8)*255
        check_img_[pad:pad_h-pad,pad:pad_w-pad] = check_img
        check_img_[pad:pad+inner_pad,:] = 255
        check_img_[:,pad:pad+inner_pad] = 255
        check_img_[-inner_pad-pad:,:] = 255
        check_img_[:,-inner_pad-pad:] = 255

        ##############
        # OCR
        x = np.expand_dims(resize_and_normailze(check_img_), axis=0)

        result2 = str(net.predict(x, classnames))
        print("[Korean License]: After Warping Result = " + result2)
#        cv2.imwrite("/root/catkin_ws/src/Korean-License-Plate-Recognition/result/" + str(video_number) + "_" + str(fcount) + "_" + str(net.predict(x, classnames)) + ".jpg", check_img_)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", default='/root/catkin_ws/src/Korean-License-Plate-Recognition/saved_models/weights_best.pb', help="path to weights file")
    parser.add_argument("-n", "--name", required=True, help="name is required")
    args = vars(parser.parse_args())
    tf.compat.v1.enable_eager_execution()

    global fcount
    fcount = 0

    net = LPRNet(len(classnames) + 1)
    net.load_weights(args["weights"])
    global video_number
    video_number = args["name"]
    print("[Korean License]: video name = " + video_number)
    rospy.init_node('ocr_node', anonymous=True)
    sub = rospy.Subscriber('ocr_sub', Image, callback)

    rospy.spin()
