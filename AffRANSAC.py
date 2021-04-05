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

class NFAclass:
    def __init__(self,VolumeActiveSet,Ndata,AffInfo=0):
        if AffInfo==0:
            Nsample = 4
        else:      
            Nsample = 2
        self.AffInfo = AffInfo
        self.Nsample = Nsample
        self.Ndata = Ndata
        self.logc_n = [self.log_n_choose_k(Ndata,k) for k in range(Ndata+1)]
        self.logc_k = [self.log_n_choose_k(k,Nsample) for k in range(Ndata+1)]                
        self.logconstant = np.log10( Ndata-Nsample )
        self.epsilon = 0.00000001
        if AffInfo == 2:
            self.dim = 8
            self.logalpha_base = np.log10( ((np.pi**4)/(4*3*2)) / VolumeActiveSet ) + np.log10( 0.5/np.pi )
        else:
            self.dim = 4
            self.logalpha_base = np.log10( ((np.pi**2)/2) / VolumeActiveSet ) + np.log10( 0.5/np.pi )

    @staticmethod
    def log_n_choose_k(n,k):
        if k>=n or k<=0:
            return 0.0
        if n-k<k:
            k = n-k
        r = 0.0
        for i in np.arange(1,k+1):#(int i = 1; i <= k; i++)
            r += np.log10(np.double(n-i+1))-np.log10(np.double(i))
        return r
        
    def compute_logNFA(self, k, dist):
        logalpha = self.logalpha_base + self.dim*np.log10(dist + self.epsilon) 
        return self.logconstant + logalpha*(k-self.Nsample)+self.logc_n[k]+self.logc_k[k]


def ORSAInliers(matches,kplistq, kplistt, H, Affnetdecomp=[],  thres = 24, nfa = None):
    goodM = []
    if np.linalg.matrix_rank(H) < 3:
        return goodM, -1
    AvDist = 0
    Hi = np.linalg.inv(H)
    if len(Affnetdecomp)==0:
        vec1, vec2 = np.zeros(shape=(4,1),dtype=np.float), np.zeros(shape=(4,1),dtype=np.float)
    else:
        vec1, vec2 = np.zeros(shape=(8,1),dtype=np.float), np.zeros(shape=(8,1),dtype=np.float)
    vec_dist = []
    vec_spatial_dist = []
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
        spatial_dist = cv2.norm(vec1,vec2)
        vec_spatial_dist.append( spatial_dist )

        if len(Affnetdecomp)>0 and spatial_dist<=thres:
            avecnet = Affnetdecomp[i][0:4]                
            avec = affine_decomp(FirstOrderApprox_Homography(H,kplistq[m.queryIdx].pt+tuple([1])), doAssert=False)[0:4]

            Affdiff = [ avec[0]/avecnet[0] if avec[0]>avecnet[0] else avecnet[0]/avec[0], 
                        AngleDiff(avec[1],avecnet[1],InRad=True), 
                        avec[2]/avecnet[2] if avec[2]>avecnet[2] else avecnet[2]/avec[2] , 
                        AngleDiff(avec[3],avecnet[3],InRad=True) ]
            vec1[4:8,0] = np.array(Affdiff)
            vec2[4:8,0] = np.array( [1.0, 0.0, 1.0, 0.0] )

        if spatial_dist>thres:   
            vec_dist.append( np.inf )
        else:
            if len(Affnetdecomp)>0:
                vec_dist.append( cv2.norm(vec1,vec2) )
            else:
                vec_dist.append( spatial_dist )
    
    vec_ordered_idx = np.argsort( vec_dist )
    vec_dist = [vec_dist[i] for i in vec_ordered_idx]
    best_nfa = np.inf
    best_k = 0 
    for k in range(2,len(matches)):
        if k<len(matches)-1 and vec_dist[k]==vec_dist[k+1]:
            continue
        if vec_spatial_dist[vec_ordered_idx[k]]>thres:
            break
        nfa_val = nfa.compute_logNFA(k-1,vec_dist[k]) 
        if nfa_val<best_nfa:
            best_nfa = nfa_val
            best_k = k
    if best_nfa<0:
        goodM = [matches[vec_ordered_idx[i]] for i in range(best_k+1)]
    return goodM, best_nfa


def Aff_RANSAC_H(img1, cvkeys1, img2, cvkeys2, cvMatches, pxl_radius = 20, Niter= 1000, AffInfo = 0, precision=24, Aq2t=None, ORSAlike=False):
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
    bestScore = [np.inf,0]
    bestMatches = []
    Ns = 2 if AffInfo>0 else 4
    if len(cvMatches)<=Ns:
        return  bestScore, bestH, bestMatches
    if ORSAlike:
        h1,w1 = np.shape(img1)
        h2,w2 = np.shape(img2)
        VolumeActiveSet = h1*w1*h2*w2
        if AffInfo==2:
            VolumeActiveSet = VolumeActiveSet*(np.pi**2)*12*12
        Ndata = len(cvMatches)
        nfa_obj = NFAclass(VolumeActiveSet,Ndata,AffInfo=AffInfo)
        def call_ORSA(*args, **kwargs):
            goodM, nfa_val = ORSAInliers(*args, **kwargs, nfa=nfa_obj)
            return goodM, [nfa_val,len(goodM)]
        find_inliers = call_ORSA
    else:
        def call_Look4Inliers(*args, **kwargs):
            goodM, AvDist = Look4Inliers(*args, **kwargs)
            return goodM, [-len(goodM),AvDist]
        find_inliers = call_Look4Inliers
    
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
                goodM, scorevec = find_inliers(cvMatches,cvkeys1, cvkeys2, H, Affnetdecomp = [], thres=precision )
            elif AffInfo==2:
                goodM, scorevec = find_inliers(cvMatches,cvkeys1, cvkeys2, H, Affnetdecomp = Affdecomp, thres=precision )
        else:
            H = HomographyFit([Xi[mi] for mi in m], Y0=[Yi[mi] for mi in m])
            goodM, scorevec = find_inliers(cvMatches,cvkeys1, cvkeys2, H, Affnetdecomp = [], thres=precision )
        
        if bestScore[0]>scorevec[0] or (bestScore[0]==scorevec[0] and bestScore[1]>scorevec[1]):
            bestScore = scorevec
            bestH = H
            bestMatches = goodM
    if ORSAlike:
        score2return = bestScore[0]
    else:
        score2return = -bestScore[0]
    return  score2return, bestH, bestMatches