import socket  
import cv2  
import threading  
import struct  
import sys
sys.path.append("./build/lib.linux-aarch64-2.7")
import mypack
import procfunc
import math
import numpy as np
import time
mypack.netInit()

if __name__ == "__main__":
    teamName = 'ICT-CAS'
    DAC = './'
    [imgDir, resultDir, timeDir, xmlDir, myXmlDir, allTimeFile] = procfunc.setupDir(DAC, teamName)
    [allImageName, imageNum] = procfunc.getImageNames(imgDir)
    batchNumDiskToDram = 1
    batchNumDramToGPU = 1
    imageReadTime = int(math.ceil(imageNum/float(batchNumDiskToDram)))
    imageProcTimeEachRead = int(math.ceil(batchNumDiskToDram/float(batchNumDramToGPU)))
    mypack.netInit()

    for i in range(int(imageReadTime)):
        imgName = imgDir + '/' + allImageName[i]
        img = cv2.imread(imgName, 1)
        print("loaded img {}".format(imgName))
        detecRec = procfunc.detectionAndTracking(img.astype(np.float32, copy=False), 1)[0]
        print " perform detection processing successfully, and the result is "+str(detecRec) 
        cv2.rectangle(img, (abs(int(detecRec[0])),abs(int(detecRec[2]))),(abs(int(detecRec[1])),abs(int(detecRec[3]))),(0,255,0),4)   
        cv2.imshow("tmp", img)    
        cv2.waitKey(1)       


