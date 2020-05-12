import cv2
import sys
sys.path.append(".")
from library import *
import random
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



def HomographyFit(X0, Y0=[], Aff=[]):
    ''' Fits an homography from coordinate correspondances and 
    if local affine info is present then we also use it to better fit
    the homography.
    Remarks:
        1. If there is no affine info (Aff=[]) then you need at least 4 correspondances.
        2. If you do provide affine infos you only need 2 correspondances.
    '''
    Affeqqs = True if len(Aff)>0 else False
    assert ( len(X0)==len(Aff) or Affeqqs==False ) and len(X0)>0
    n = len(X0)
    if len(X0[0])==3:
        Xi = np.array( [[X0[i][0]/X0[i][2], X0[i][1]/X0[i][2]]  for i in range(n)] )
    else:
        Xi = np.array( [[X0[i][0], X0[i][1]]  for i in range(n)] )
    
    if Affeqqs:
        eqN = 6
        Yi = np.array( [np.matmul(Aff[i][0:2,0:2],Xi[i]) + Aff[i][:,2] for i in range(n)] )
    else:
        eqN = 2
        if len(Y0[0])==3:
            Yi = np.array( [[Y0[i][0]/Y0[i][2], Y0[i][1]/Y0[i][2]]  for i in range(n)] )
        else:
            Yi = np.array( [[Y0[i][0], Y0[i][1]]  for i in range(n)] ) 

    A = np.zeros((eqN*n,9),dtype=np.float)
    for i in range(n):
        # Coordinates constraints
        j = eqN*i
        A[j,0] = Xi[i,0]
        A[j,1] = Xi[i,1]        
        A[j,2] = 1.0
        A[j,6] = - Yi[i,0] * Xi[i,0]
        A[j,7] = - Yi[i,0] * Xi[i,1]
        A[j,8] = - Yi[i,0]

        j = eqN*i + 1
        A[j,3] = Xi[i,0]
        A[j,4] = Xi[i,1]
        A[j,5] = 1.0
        A[j,6] = - Yi[i,1] * Xi[i,0]
        A[j,7] = - Yi[i,1] * Xi[i,1]
        A[j,8] = - Yi[i,1]
        
        if Affeqqs:
            AA = Aff[i][0:2,0:2]

            # Affine constraints
            j = eqN*i + 2
            A[j,0] = 1.0
            A[j,6] = - Yi[i,0] - AA[0,0] * Xi[i,0]
            A[j,7] = - AA[0,0] * Xi[i,1]
            A[j,8] = - AA[0,0]        

            j = eqN*i + 3
            A[j,1] = 1.0
            A[j,6] = - AA[0,1] * Xi[i,0]
            A[j,7] = - Yi[i,0] - AA[0,1] * Xi[i,1]
            A[j,8] = - AA[0,1]

            j = eqN*i + 4
            A[j,3] = 1.0
            A[j,6] = - Yi[i,1] - AA[1,0] * Xi[i,0]
            A[j,7] = - AA[1,0] * Xi[i,1]
            A[j,8] = - AA[1,0]

            j = eqN*i + 5
            A[j,4] = 1.0
            A[j,6] = - AA[1,1] * Xi[i,0]
            A[j,7] = - Yi[i,1] - AA[1,1] * Xi[i,1]
            A[j,8] = - AA[1,1]
        

    _, _, v = np.linalg.svd(A)
    h = np.reshape(v[8], (3, 3))
    if h.item(8)!=0.0:
        h = (1/h.item(8)) * h
    return h


def Look4Inliers(matches,kplistq, kplistt, H, Affnetdecomp=[],  thres = 24):
        goodM = []
        if np.linalg.matrix_rank(H) < 3:
            return goodM, -1
        affdiffthres = np.array( [2.0, np.pi/4, 2.0, np.pi/4] )
        # affdiffthres = np.array( [np.inf, np.inf, np.inf, np.inf] )
        AvDist = 0
        Hi = np.linalg.inv(H)
        vec1, vec2 = np.zeros(shape=(4,1),dtype=np.float), np.zeros(shape=(4,1),dtype=np.float)
        for i in range(len(matches)):
            m = matches[i]
            x = kplistq[m.queryIdx].pt + tuple([1])
            x = np.array(x).reshape(3,1)
            Hx = np.matmul(H, x)
            Hx = Hx/Hx[2]
            
            y = kplistt[m.trainIdx].pt + tuple([1])
            y = np.array(y).reshape(3,1)
            Hiy = np.matmul(Hi, y)
            Hiy = Hiy/Hiy[2]

            vec1[0:2,0] = Hx[0:2,0]
            vec1[2:4,0] = x[0:2,0]
            vec2[0:2,0] = y[0:2,0]
            vec2[2:4,0] = Hiy[0:2,0]

            thisdist = cv2.norm(vec1,vec2)
            if len(Affnetdecomp)==0 and thisdist <= thres:
                goodM.append(m)
                AvDist += thisdist
            elif len(Affnetdecomp)>0 and thisdist <= thres:
                avecnet = Affnetdecomp[i][0:4]                
                avec = affine_decomp(FirstOrderApprox_Homography(H,kplistq[m.queryIdx].pt+tuple([1])), doAssert=False)[0:4]

                Affdiff = [ avec[0]/avecnet[0] if avec[0]>avecnet[0] else avecnet[0]/avec[0], 
                            AngleDiff(avec[1],avecnet[1],InRad=True), 
                            avec[2]/avecnet[2] if avec[2]>avecnet[2] else avecnet[2]/avec[2] , 
                            AngleDiff(avec[3],avecnet[3],InRad=True) ]

                if (Affdiff<affdiffthres).all():
                    goodM.append(m)
                    AvDist += thisdist              
        if len(goodM)>0:                
            AvDist = AvDist/len(goodM)    
        else:
            AvDist = -1    
        return goodM, AvDist

def Aff_RANSAC_H(img1, cvkeys1, img2, cvkeys2, cvMatches, pxl_radius = 20, Niter= 1000, AffInfo = 0, precision=24, Aq2t=None):
    '''
    AffInfo == 0 - RANSAC Vanilla
    AffInfo == 1 - Fit Homography to affine info + Classic Validation
    AffInfo == 2 - Fit Homography to affine info + Affine Validation
    '''    
    def compute_image_patches(img, cvPts):
        ''' 
        Compute AID descriptors for:
            img - input image
            cvPts - a list of opencv keypoints
        '''
        
        maxoctaves = np.max( [unpackSIFTOctave(kp)[0] for kp in cvPts] )
        pyr1 = buildGaussianPyramid( img, maxoctaves+2 )
        patches1, A_list1, Ai_list1 = ComputePatches(cvPts,pyr1, border_mode=cv2.BORDER_REFLECT)
        ## KPlist1, _ = Filter_Affine_In_Rect(KPlist1,A1,[0,0],[w,h],isSIFT=True)
        bP = np.zeros( shape = tuple([len(patches1)])+tuple(np.shape(patches1[0]))+tuple([1]), dtype=np.float32)
        for k in range(0,len(patches1)):
            bP[k,:,:,0] = patches1[k][:,:]/255.0
            # WriteImgKeys(patches1[k], [], 'p1/'+str(k)+'.png')
            # WriteImgKeys(img, [cvPts[k]], 'im1/'+str(k)+'.png')
        return patches1


    if AffInfo>0:
        w, h = 60, 60 
        patches1 = compute_image_patches(img1,cvkeys1)
        patches2 = compute_image_patches(img2,cvkeys2)
        
        if Aq2t is None:
            bP = np.zeros(shape=tuple([len(cvkeys1),60,60,2]), dtype = np.float32)
        Akp1, Akp2 = [], []
        keys2seek = []
        for n in range(len(cvMatches)):
            idx1 = cvMatches[n].queryIdx
            idx2 = cvMatches[n].trainIdx
            if Aq2t is None:
                bP[n,:,:,:] = np.dstack((patches1[idx1]/255.0, patches2[idx2]/255.0))
            Akp1.append(kp2LocalAffine(cvkeys1[idx1]))
            Akp2.append(kp2LocalAffine(cvkeys2[idx2]))
            
            ## degub keyponits to keep track in images
            InPatchKeys2seek = [cv2.KeyPoint(x = w/2+pxl_radius*i - pxl_radius/2, y = h/2 +pxl_radius*j - pxl_radius/2,
                _size =  cvkeys1[idx1].size, 
                _angle =  0.0,
                _response =  cvkeys1[idx1].response, _octave =  cvkeys1[idx1].octave,
                _class_id =  i*2+j) for i in range(0,2) for j in range(0,2)]
            InPatchKeys2seek.append(cv2.KeyPoint(x = w/2, y = h/2,
                _size =  cvkeys1[idx1].size, 
                _angle =  0.0,
                _response =  cvkeys1[idx1].response, _octave =  cvkeys1[idx1].octave,
                _class_id =  -1))
            temp = AffineKPcoor(InPatchKeys2seek, cv2.invertAffineTransform(Akp1[n]))
            keys2seek.append( temp ) 
        
        if Aq2t is None:
            global graph
            with graph.as_default():
                bEsti =LOCATEmodel.layers[2].predict(bP)
        GA = GenAffine("", DryRun=True)
        
        if Aq2t is None:
            Affmaps = []
        else:
            assert len(Aq2t)==len(cvMatches)
            Affmaps = Aq2t
        Affdecomp = []
        Xi = []
        for n in range(len(cvMatches)):
            if Aq2t is None:
                evec = bEsti[n,:]
                A_p1_to_p2 = cv2.invertAffineTransform( GA.AffineFromNormalizedVector(evec) )
                A_query_to_p2 = ComposeAffineMaps(A_p1_to_p2,Akp1[n])
                # A_p1_to_target = ComposeAffineMaps( cv2.invertAffineTransform(Akp2[n]), A_p1_to_p2 )
                A_query_to_target = ComposeAffineMaps( cv2.invertAffineTransform(Akp2[n]), A_query_to_p2 )
                Affmaps.append( A_query_to_target )
                Affdecomp.append( affine_decomp(A_query_to_target) )
            else:
                Affdecomp.append( affine_decomp(Aq2t[n]) )
            Xi.append( np.array( cvkeys1[cvMatches[n].queryIdx].pt+tuple([1]) ) )

            # keys4p1 = AffineKPcoor(keys2seek[n], Akp1[n])
            # keys4p2 = AffineKPcoor(keys2seek[n], A_query_to_p2)
            # keys4target = AffineKPcoor(keys2seek[n], A_query_to_target)

            idx1 = cvMatches[n].queryIdx
            idx2 = cvMatches[n].trainIdx
            # WriteImgKeys(patches1[idx1], keys4p1, 'p1/'+str(n)+'.png' )
            # WriteImgKeys(patches2[idx2], keys4p2, 'p2/'+str(n)+'.png' )
            # WriteImgKeys(img1, keys2seek[n], 'im1/'+str(n)+'.png' )
            # WriteImgKeys(img2, keys4target, 'im2/'+str(n)+'.png' )
            
        # update cvkeys2 with refined position
        Yi = np.array( [np.matmul(Affmaps[n][0:2,0:2],Xi[n][0:2]/Xi[n][2]) + Affmaps[n][:,2] for n in range(len(cvMatches))] )
        for n in range(len(cvMatches)):    
            cvkeys2[cvMatches[n].trainIdx].pt = tuple(Yi[n])
    else:
        Xi = []
        Yi = []
        for n in range(len(cvMatches)):
            Xi.append( np.array( cvkeys1[cvMatches[n].queryIdx].pt+tuple([1]) ) )
            Yi.append( np.array( cvkeys2[cvMatches[n].trainIdx].pt+tuple([1]) ) )



    # RANSAC
    bestH = []
    bestCount = 0
    bestMatches = []
    if len(cvMatches)<4:
        return bestCount, bestH, bestMatches
    
    Ns = 2 if AffInfo>0 else 4
    for i in range(Niter):
        m = -1*np.ones(Ns,np.int)
        for j in range(Ns):
            m1 = np.random.randint(0,len(cvMatches))
            while m1 in m:
                m1 = np.random.randint(0,len(cvMatches))
            m[j] = m1
        if AffInfo>0:
            H = HomographyFit([Xi[mi] for mi in m], Aff=[Affmaps[mi] for mi in m])
            if AffInfo==1:
                goodM, _ = Look4Inliers(cvMatches,cvkeys1, cvkeys2, H, Affnetdecomp = [], thres=precision )
            elif AffInfo==2:
                goodM, _ = Look4Inliers(cvMatches,cvkeys1, cvkeys2, H, Affnetdecomp = Affdecomp, thres=precision )
        else:
            H = HomographyFit([Xi[mi] for mi in m], Y0=[Yi[mi] for mi in m])
            goodM, _ = Look4Inliers(cvMatches,cvkeys1, cvkeys2, H, Affnetdecomp = [], thres=precision )
        
        if bestCount<len(goodM):
            bestCount = len(goodM)
            bestH = H
            bestMatches = goodM
    return  bestCount, bestH, bestMatches