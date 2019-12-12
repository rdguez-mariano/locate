import cv2
import sys
sys.path.append(".")
from library import *
from libLocalDesc import *

from matplotlib import pyplot as plt
plt.switch_backend('agg')


def WriteImgKeys(img, keys, pathname, Flag=2):
        colors=( (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255) )
        patch = cv2.cvtColor( img.astype(np.uint8), cv2.COLOR_GRAY2RGB )
        if len(keys)>len(colors):
            patch=cv2.drawKeypoints(patch,keys,patch, flags=Flag)
        else:
            for n in range(np.min([len(colors),len(keys)])):
                patch=cv2.drawKeypoints(patch,keys[n:n+1],patch, color=colors[n] ,flags=Flag)
        cv2.imwrite('temp/'+pathname,patch)


def CreateOnImage_cvPts(listPts):
    '''
    Create a list of opencv keypoints from a 
    list of points [(x1,y1), (x2,y2), ..., (xn,yn)]
    '''
    return [cv2.KeyPoint(x = pt[0], y = pt[1],
            _size = 2.0, _angle = 0.0,
            _response = 1.0, _octave = packSIFTOctave(-1,0),
            _class_id = 0) for pt in listPts]

def compute_image_AID(img, cvPts):    
    maxoctaves = 5
    pyr1 = buildGaussianPyramid( img, maxoctaves+2 )
    patches1, A_list1, Ai_list1 = ComputePatches(cvPts,pyr1, border_mode=cv2.BORDER_REFLECT)
    return patches1


def batch_hamming(a,b):
        res = np.zeros(np.shape(a)[0],dtype=np.int)
        for i in range(0,np.shape(a)[0]):
            res[i] = np.shape(a)[1] - np.sum(1*(np.logical_xor(a[i,:],b[i,:])))
        return res


def WriteTangentsInTarget(img1, img2, pxl_radius = 20):
    im1 = img1.copy()

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    w, h = 60, 60
    mask = np.zeros((h, w), np.uint8)
    mask[:] = 1.0


    total, good_HC, kplistq, kplistt, H_aid, _, _ = siftAID(img1,img2, MatchingThres = 4000, Simi = 'SignProx', Visual=False, GFilter='Aff_H-2')    
    good_HC = OnlyUniqueMatches(good_HC,kplistq,kplistt,SpatialThres=60)

    cvkeys1 = [kplistq[good_HC[i].queryIdx] for i in range(len(good_HC))]
    cvkeys2 = [kplistt[good_HC[i].trainIdx] for i in range(len(good_HC))]
    patches1 =compute_image_AID(img1, cvkeys1)    
    patches2 =compute_image_AID(img2, cvkeys2)

    bP = np.zeros(shape=tuple([len(cvkeys1),60,60,2]), dtype = np.float32)
    Akp1, Akp2 = [], []
    keys2seek = []
    for n in range(len(cvkeys1)):
        bP[n,:,:,:] = np.dstack((patches1[n]/255.0, patches2[n]/255.0))
        Akp1.append(kp2LocalAffine(cvkeys1[n]))
        Akp2.append(kp2LocalAffine(cvkeys2[n]))
        
        # degub keyponits to keep track in images
        InPatchKeys2seek = [cv2.KeyPoint(x = w/2+pxl_radius*i - pxl_radius/2, y = h/2 +pxl_radius*j - pxl_radius/2,
            _size =  2.0, 
            _angle =  0.0,
            _response =  cvkeys1[n].response, _octave =  packSIFTOctave(-1,0),
            _class_id =  i*2+j) for i in range(0,2) for j in range(0,2)]
        InPatchKeys2seek.append(cv2.KeyPoint(x = w/2, y = h/2,
            _size =  2.0, 
            _angle =  0.0,
            _response =  cvkeys1[n].response, _octave =  packSIFTOctave(-1,0),
            _class_id =  -1))
        temp = AffineKPcoor(InPatchKeys2seek, cv2.invertAffineTransform(Akp1[n]))
        keys2seek.append( temp ) 

    bEsti =LOCATEmodel.layers[2].predict(bP)
    GA = GenAffine("", DryRun=True)

    Affmaps = []
    imgs = []
    masks = []
    kps1, kps2 = [], []
    cornersT, cornersQ = [], []
    SquarePatch = SquareOrderedPts(h,w,CV=True)
    for n in range(len(cvkeys1)):
        o1, l1, _ = unpackSIFTOctave(cvkeys1[n])
        o2, l2, _ = unpackSIFTOctave(cvkeys2[n])
        if o1>-1 or o2>-1:
            continue
        
        kps1.append( cvkeys1[n] )
        kps2.append( cvkeys2[n] )
        evec = bEsti[n,:]
        A_p1_to_p2 = cv2.invertAffineTransform( GA.AffineFromNormalizedVector(evec) )        
        A_query_to_p2 = ComposeAffineMaps(A_p1_to_p2,Akp1[n])
        A_p1_to_target = ComposeAffineMaps( cv2.invertAffineTransform(Akp2[n]), A_p1_to_p2 )
        A_query_to_target = ComposeAffineMaps( cv2.invertAffineTransform(Akp2[n]), A_query_to_p2 )
        Affmaps.append( A_query_to_target )

        keys4p1 = AffineKPcoor(keys2seek[n], Akp1[n])
        keys4p2 = AffineKPcoor(keys2seek[n], A_query_to_p2)
        keys4target = AffineKPcoor(keys2seek[n], A_query_to_target)
        
        cornersQ.append( AffineKPcoor(SquarePatch, cv2.invertAffineTransform(Akp1[n])) )
        cornersT.append( AffineKPcoor(SquarePatch, A_p1_to_target) )

        WriteImgKeys(patches1[n], keys4p1, 'p1/'+str(n)+'.png' )
        WriteImgKeys(patches2[n], keys4p2, 'p2/'+str(n)+'.png' )
        WriteImgKeys(img1, keys2seek[n], 'im1/'+str(n)+'.png' )
        WriteImgKeys(img2, keys4target, 'im2/'+str(n)+'.png' )

        patch = cv2.warpAffine(img1, Akp1[n], (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        img = cv2.warpAffine(patch, A_p1_to_target, (w2, h2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        imgs.append( img )
        WriteImgKeys(img, keys4target, 'p1init/'+str(n)+'.png' )
        masks.append( cv2.warpAffine(mask, A_p1_to_target, (w2, h2), flags=cv2.INTER_LINEAR) )

    

    masks = np.sum(masks,axis=0)
    backimg = 0.5*img2
    backimg[masks>0] = 0.0
    masks[masks==0] = 1.0
    img = np.sum(imgs,axis=0)/masks + backimg
    for n in range(len(cornersT)):
        pts = np.array([np.array(kp.pt) for kp in cornersT[n]], np.int32)
        pts = pts.reshape((-1,1,2))
        img = cv2.polylines(img,[pts],True,(255,255,255))

        pts = np.array([np.array(kp.pt) for kp in cornersQ[n]], np.int32)
        pts = pts.reshape((-1,1,2))
        img1 = cv2.polylines(img1,[pts],True,(255,255,255))

    img3 = cv2.drawMatches(im1,kps1,(0.5*img2).astype(np.uint8),kps2,[cv2.DMatch(i,i,1.0) for i in range(len(kps1))], None,flags=2)
    cv2.imwrite('./temp/teaser_matches.png',img3)
    img3 = cv2.drawMatches(img1,kps1,img.astype(np.uint8),kps2,[cv2.DMatch(i,i,1.0) for i in range(len(kps1))], None,flags=2)
    cv2.imwrite('./temp/teaser_adv.png',img3)



img1 = cv2.cvtColor(cv2.imread('./acc-test/adam.1.png'),cv2.COLOR_BGR2GRAY) # queryImage
img2 = cv2.cvtColor(cv2.imread('./acc-test/adam.2.png'),cv2.COLOR_BGR2GRAY) # queryImage

WriteTangentsInTarget(img1, img2)