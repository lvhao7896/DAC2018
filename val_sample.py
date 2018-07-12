import numpy as np
import sys
import os
import xml.etree.ElementTree as ET
import cv2

class bbox():
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.width = ymax - ymin
        self.height = xmax - xmin
        
    def overlap_width(self, box):
        maxl = self.xmin if self.xmin > box.xmin else box.xmin
        minr = self.xmax if self.xmax < box.xmax else box.xmax
        return minr - maxl
        
    def overlap_height(self, box):
        maxt = self.ymin if self.ymin > box.ymin else box.ymin
        minb = self.ymax if self.ymax < box.ymax else box.ymax
        return minb - maxt
        
    def box_intersection(self, box):
        w = self.overlap_width(box)
        h = self.overlap_height(box)
        w = w if w > 0 else 0
        h = h if h > 0 else 0
        area = w * h
        # print("intersection {}".format(area))
        return area
        
    def box_union(self, box):
        i = self.box_intersection(box)
        u = self.height * self.width + box.height * box.width - i
        # print("union {}".format(u))
        return u
        
    def iou(self, box):
        return self.box_intersection(box) / self.box_union(box)
        
    def print_box(self):
        print("({} {}) {} X {}".format(self.xmin, self.ymin, self.width, self.height))



label_dir = './images'
pred_dir='./result/xml/ICT-CAS'
files = os.listdir(label_dir)
label_xmls = []
sum_iou = 0.
count = 0
for i in files:
    if 'xml' in i:
        label_xmls.append(i)
for i in label_xmls:
    count += 1
    pred_xml = pred_dir + '/' + i
    predxmltree = ET.parse(pred_xml)
    obj = predxmltree.findall('object')
    if obj == None:
        print("error : no obj in {}", pred_xml)
    pred_bbox = obj[0].findall('bndbox')[0]
    xmin = float(pred_bbox.find('xmin').text)
    xmax = float(pred_bbox.find('xmax').text)
    ymin = float(pred_bbox.find('ymin').text)
    ymax = float(pred_bbox.find('ymax').text)
    pred_bbox = bbox(xmin, ymin, xmax, ymax)
    gt_xml = label_dir + '/' + i
    gt_xmltree = ET.parse(gt_xml)
    gt = gt_xmltree.findall('object')
    if gt == None:
        print("error : no obj in {}".format(gt_xml))
    # gt_bbox = gt[0].findall('bndbox')[0]
    gt_bbox = gt[0].findall('bndbox')[0]
    gt_xmin = float(gt_bbox.find('xmin').text)
    gt_xmax = float(gt_bbox.find('xmax').text)
    gt_ymin = float(gt_bbox.find('ymin').text)
    gt_ymax = float(gt_bbox.find('ymax').text)
    gt_bbox = bbox(gt_xmin, gt_ymin, gt_xmax, gt_ymax)
    cur_iou = gt_bbox.iou(pred_bbox)
    sum_iou += cur_iou
    print("index : {} {} \n\t avg_iou: {} cur_iou: {}".format(count, i,
         sum_iou/count, cur_iou))

