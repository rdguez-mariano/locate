#!/usr/bin/env python
import cv2
from libLocalDesc import *
from library import opt

img1 = cv2.cvtColor(cv2.imread(opt.im1),cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(cv2.imread(opt.im2),cv2.COLOR_BGR2GRAY)

# HessAff + AID + USAC Homography

if opt.detector == 'HessAff':
    if opt.descriptor == 'AID':
        _, good_HC, _,_,_, ET_KP, ET_M = HessAffAID(img1,img2)
    elif opt.descriptor == 'HardNet':
        _, good_HC, _,_,_, ET_KP, ET_M = HessAffNet_HardNet(img1,img2)
elif opt.detector == 'SIFT':
    if opt.descriptor == 'AID':
        _, good_HC, KPlist1, KPlist2, H_AID, ET_KP, ET_M = siftAID(img1,img2)
    elif opt.descriptor == 'HardNet':
        _, good_HC, KPlist1, KPlist2, H_sift, ET_KP, ET_M = SIFT_AffNet_HardNet(img1,img2)

print("FilteredMatches = %d, KeypointsTime = %3.3f, MatchingTime = %3.3f" %(len(good_HC),ET_KP,ET_M))