import cv2
import sys
sys.path.append(".")
from library import *
from libLocalDesc import *
from acc_test_library import *
from matplotlib import pyplot as plt
plt.switch_backend('agg')


ds = LoadDatasets()

def CorrectMatches(matches,kplistq, kplistt, H, thres = 10):
    goodM = []
    AvDist = 0
    for m in matches:
        x = kplistq[m.queryIdx].pt + tuple([1])
        x = np.array(x).reshape(3,1)
        Hx = np.matmul(H, x)
        Hx = Hx/Hx[2]
        
        y =kplistt[m.trainIdx].pt
        thisdist = cv2.norm(Hx[0:2],y)
        if  thisdist <= thres:
            goodM.append(m)
            AvDist += thisdist
    if len(goodM)>0:                
        AvDist = AvDist/len(goodM)    
    else:
        AvDist = -1    
    return goodM, AvDist



# p = ds[0].datapairs[11]
p = ds[0].datapairs[0]

total, good_HC, kplistq, kplistt, H, ET_KP, ET_M = siftAID(p.query,p.target, GetAllMatches = True, MatchingThres = 4000, Simi = 'SignProx',  GFilter='USAC_H')
cmHC, _ = CorrectMatches(good_HC,kplistq, kplistt, p.Tmatrix )
cmT, _ = CorrectMatches(total,kplistq, kplistt, p.Tmatrix )


img1 = cv2.drawMatches(p.query, kplistq,p.target, kplistt, total, None,flags=2,matchColor=(0, 255, 0))
cv2.imwrite('./temp/siftaid_total_matches.png',img1)
img1 = cv2.drawMatches(p.query, kplistq,p.target, kplistt, cmHC, None,flags=2,matchColor=(0, 255, 0))
cv2.imwrite('./temp/siftaid_cmHC_matches.png',img1)
img1 = cv2.drawMatches(p.query, kplistq,p.target, kplistt, cmT, None,flags=2,matchColor=(0, 255, 0))
cv2.imwrite('./temp/siftaid_cmT_matches.png',img1)


total, good_HC, kplistq, kplistt, H, ET_KP, ET_M = siftAID(p.query,p.target, MatchingThres = 4000, Simi='SignProx', knn_num = 1, GFilter='USAC_H', RegionGrowingIters=4)
cmHC, _ = CorrectMatches(good_HC,kplistq, kplistt, p.Tmatrix )
cmT, _ = CorrectMatches(total,kplistq, kplistt, p.Tmatrix )

img1 = cv2.drawMatches(p.query, kplistq,p.target, kplistt, total, None,flags=2,matchColor=(0, 255, 0))
cv2.imwrite('./temp/Guided_siftaid_total_matches.png',img1)
img1 = cv2.drawMatches(p.query, kplistq,p.target, kplistt, cmHC, None,flags=2,matchColor=(0, 255, 0))
cv2.imwrite('./temp/Guided_siftaid_cmHC_matches.png',img1)
img1 = cv2.drawMatches(p.query, kplistq,p.target, kplistt, cmT, None,flags=2,matchColor=(0, 255, 0))
cv2.imwrite('./temp/Guided_siftaid_cmT_matches.png',img1)