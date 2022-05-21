# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 14:42:11 2021

@author: Toqa Alaa
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2


def detect_shapes(path):

    img = cv2.imread(path)
    #cv2.imshow('img', img)

      
    img_contour = img.copy()
    img_blur = cv2.GaussianBlur(img, (7, 7), 1)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray',img_gray)
    cv2.waitKey(0)
    ret, thrash = cv2.threshold(img_gray, 60 , 500, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
           

    def find_shape(approx):
        if len(approx) == 3:
            s = "Triangle"
        elif len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ar = w / float(h)
            s = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        elif len(approx) == 5:
            s = "Pentagon"
        elif len(approx) == 8:
            s = "Octagon"
        else:
            s = "Circle"
        return s
    
    
    for cnt in contours:
        epsilon = 0.013 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(img_contour, [approx], 0, (255,0,255), 2)
        shape = find_shape(approx)  
        M = cv2.moments(cnt)
        if M['m00'] != 0.0:
            xc = int(M['m10']/M['m00'])
            yc = int(M['m01']/M['m00'])
        cv2.putText(img, shape , (xc-25, yc+57), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
    
    #cv2.imshow('contour', img_contour)
    cv2.waitKey(0)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    
detect_shapes('geo.png')
    
    
