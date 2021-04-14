#!/usr/bin/env python
import cv2
from libLocalDesc import *
from library import opt

img1 = cv2.cvtColor(cv2.imread(opt.im1),cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(cv2.imread(opt.im2),cv2.COLOR_BGR2GRAY)

if opt.detector == 'HessAff':
    if opt.descriptor == 'AID':
        allmatches, good_HC, KPlist1, KPlist2, H_matrix, ET_KP, ET_M = HessAffAID(img1,img2, affmaps_method=opt.affmaps, EuPrecision=opt.precision, ransac_iters=opt.ransac_iters)
    elif opt.descriptor == 'HardNet':
        allmatches, good_HC, KPlist1, KPlist2, H_matrix, ET_KP, ET_M = HessAff_HardNet(img1,img2,HessAffNet=False, affmaps_method=opt.affmaps, EuPrecision=opt.precision, ransac_iters=opt.ransac_iters)
elif opt.detector == 'SIFT':
    if opt.descriptor == 'AID':
        allmatches, good_HC, KPlist1, KPlist2, H_matrix, ET_KP, ET_M = siftAID(img1,img2, affmaps_method=opt.affmaps, EuPrecision=opt.precision, ransac_iters=opt.ransac_iters)
    elif opt.descriptor == 'HardNet':
        allmatches, good_HC, KPlist1, KPlist2, H_matrix, ET_KP, ET_M = SIFT_AffNet_HardNet(img1,img2,AffNetBeforeDesc=False,affmaps_method=opt.affmaps, EuPrecision=opt.precision, ransac_iters=opt.ransac_iters)

img3 = cv2.drawMatches(img1,KPlist1,img2,KPlist2,allmatches, None,flags=2)
warped = 0.3*img2
if len(good_HC)>0:
    print("Estimated Homography matrix :\n",H_matrix)
    print("%i matches out of %i are in consensus with the above homography" % (len(good_HC),len(allmatches)))
    h, w = img2.shape[:2]
    warped = 0.7*cv2.warpPerspective(img1, H_matrix,(w, h)) + warped
else:
    print("No consensual matches have been found !")


cv2.imwrite(opt.workdir+'all_matches.png',img3)
cv2.imwrite(opt.workdir+'queryontarget.png',warped)