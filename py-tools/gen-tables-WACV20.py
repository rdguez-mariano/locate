import cv2
import sys
sys.path.append(".")
from library import *
from acc_test_library import *
from libLocalDesc import *
from tqdm import tqdm

CosProxThres = 0.4
SignAlingThres = 4000
GeoFilter = 'USAC_H'
SameKPthres = 4


from AffRANSAC import HomographyFit, Look4Inliers
def LaunchAndRecord_Aff_RANSAC_H(MS, p, kplistq, kplistt, total, AffInfo = 0, Aq2t=None, BigIters=10):
    '''
    AffInfo == 0 - RANSAC Vanilla
    AffInfo == 1 - Fit Homography to affine info + Classic Validation
    AffInfo == 2 - Fit Homography to affine info + Affine Validation
    '''    
    img1 = p.query
    img2 = p.target
    cvkeys1 = features_deepcopy(kplistq)
    cvkeys2 = features_deepcopy(kplistt)
    cvMatches = matches_deepcopy(total)
    pxl_radius = 20
    Niter= 1000
    precision=24

    def compute_image_patches(img, cvPts):
        ''' 
        Compute AID descriptors for:
            img - input image
            cvPts - a list of opencv keypoints
        '''
        
        maxoctaves = np.max( [unpackSIFTOctave(kp)[0] for kp in cvPts] )
        pyr1 = buildGaussianPyramid( img, maxoctaves+2 )
        patches1, A_list1, Ai_list1 = ComputePatches(cvPts,pyr1, border_mode=cv2.BORDER_REFLECT)
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


    for bigiter in range(BigIters):
        # RANSAC
        bestH = []
        bestCount = 0
        bestMatches = []
        if len(cvMatches)>=4:
            Ns = 2 if AffInfo>0 else 4
            for i in range(Niter):
                m = -1*np.ones(Ns,np.int)
                for j in range(Ns):
                    m1 = np.random.randint(0,len(cvMatches))
                    while m1 in m:
                        m1 = np.random.randint(0,len(cvMatches))
                    m[j] = m1
                if AffInfo>0:
                    # print('Affine Info', Ns)
                    H = HomographyFit([Xi[mi] for mi in m], Aff=[Affmaps[mi] for mi in m])
                    if AffInfo==1:
                        goodM, _ = Look4Inliers(cvMatches,cvkeys1, cvkeys2, H, Affnetdecomp = [], thres=precision )
                    elif AffInfo==2:
                        goodM, _ = Look4Inliers(cvMatches,cvkeys1, cvkeys2, H, Affnetdecomp = Affdecomp, thres=precision )
                else:
                    # print('No affine Info', Ns)
                    H = HomographyFit([Xi[mi] for mi in m], Y0=[Yi[mi] for mi in m])
                    goodM, _ = Look4Inliers(cvMatches,cvkeys1, cvkeys2, H, Affnetdecomp = [], thres=precision )

                if bestCount<len(goodM):
                    bestCount = len(goodM)
                    bestH = H
                    bestMatches = goodM
            
        cmHC, AvDist = CorrectMatches(bestMatches,kplistq, kplistt, p.Tmatrix )
        cmT, _ = CorrectMatches(total,kplistq, kplistt, p.Tmatrix )
        MS.AddInfo(cmHC, bestMatches, total, AvDist, p.pair_name)
    MS.EndOfImgTreatement()



def ModifiedSIFT_AffNet_HardNet(img1,img2, MatchingThres = 0.8, knn_num = 2):
    # find the keypoints with SIFT
    start_time = time.time()
    KPlist1, sift_des1 = ComputeSIFTKeypoints(img1, Desc = True)
    KPlist2, sift_des2 = ComputeSIFTKeypoints(img2, Desc = True)
    Identity = np.float32([[1, 0, 0], [0, 1, 0]])
    h, w = img1.shape[:2]
    KPlist1, sift_des1, temp = Filter_Affine_In_Rect(KPlist1,Identity,[0,0],[w,h], desc_list = sift_des1)
    h, w = img2.shape[:2]
    KPlist2, sift_des2, temp = Filter_Affine_In_Rect(KPlist2,Identity,[0,0],[w,h], desc_list = sift_des2)

    maxoctaves = 4
    pyr1 = buildGaussianPyramid( img1, maxoctaves + 2 )
    pyr2 = buildGaussianPyramid( img2, maxoctaves + 2 )

    patches1, A_list1, Ai_list1 = ComputePatches(KPlist1,pyr1)
    patches2, A_list2, Ai_list2 = ComputePatches(KPlist2,pyr2)

    bP = np.zeros( shape = tuple([len(patches1)])+tuple(np.shape(patches1[0]))+tuple([1]), dtype=np.float32)
    for k in range(0,len(patches1)):
        bP[k,:,:,0] = patches1[k][:,:]
    emb_1, bP_Alist1 = AffNetHardNet_describe(bP)

    bP = np.zeros( shape=tuple([len(patches2)])+tuple(np.shape(patches2[0]))+tuple([1]), dtype=np.float32)
    for k in range(0,len(patches2)):
        bP[k,:,:,0] = patches2[k][:,:]
    emb_2, bP_Alist2 = AffNetHardNet_describe(bP)
    ET_KP = time.time() - start_time

    bf = cv2.BFMatcher()
    start_time = time.time()
    sift_matches = bf.knnMatch(emb_1,emb_2, k=knn_num)
    ET_M = time.time() - start_time


    lda = CPPbridge('./build/libDA.so')
    # Apply ratio test
    sift_all = []
    if knn_num==2:
        for m,n in sift_matches:
            if m.distance < MatchingThres*n.distance:
                sift_all.append(m)
    elif knn_num==1:
        for m in sift_matches:
            if m[0].distance <= MatchingThres:
                sift_all.append(m[0])

    sift_all = OnlyUniqueMatches(sift_all,KPlist1,KPlist2,SpatialThres=SameKPthres)      
    
    # Affine maps from query to target
    Aq2t_list = []
    for m in sift_all:
        Aq2t = ComposeAffineMaps( bP_Alist2[m.trainIdx], cv2.invertAffineTransform(bP_Alist1[m.queryIdx]) )
        Aq2t = ComposeAffineMaps( Ai_list2[m.trainIdx], Aq2t )
        Aq2t = ComposeAffineMaps( Aq2t, A_list1[m.queryIdx] )
        Aq2t_list.append( Aq2t )
    return KPlist1, KPlist2, sift_all, Aq2t_list, pyr1, pyr2


def Modified_siftAID(img1,img2, MatchingThres = math.inf, Simi='SignProx', knn_num = 1):
    if Simi=='CosProx':
        FastCode = 0
    elif Simi=='SignProx':
        FastCode = 1
    elif Simi=='SignProxKNN':
        FastCode = 2
    else:
        print('Wrong similarity choice for AI-SIFT !!!')
        exit()

    # find the keypoints with SIFT
    start_time = time.time()
    KPlist1, sift_des1 = ComputeSIFTKeypoints(img1, Desc = True)
    KPlist2, sift_des2 = ComputeSIFTKeypoints(img2, Desc = True)
    Identity = np.float32([[1, 0, 0], [0, 1, 0]])
    h, w = img1.shape[:2]
    KPlist1, sift_des1, temp = Filter_Affine_In_Rect(KPlist1,Identity,[0,0],[w,h], desc_list = sift_des1)
    h, w = img2.shape[:2]
    KPlist2, sift_des2, temp = Filter_Affine_In_Rect(KPlist2,Identity,[0,0],[w,h], desc_list = sift_des2)

    maxoctaves = 4
    pyr1 = buildGaussianPyramid( img1, maxoctaves + 2 )
    pyr2 = buildGaussianPyramid( img2, maxoctaves + 2 )

    patches1, A_list1, Ai_list1 = ComputePatches(KPlist1,pyr1)
    patches2, A_list2, Ai_list2 = ComputePatches(KPlist2,pyr2)

    KPlist1, patches1 = FilterOutUselessKPs(KPlist1, patches1)
    KPlist2, patches2 = FilterOutUselessKPs(KPlist2, patches2)


    bP = np.zeros( shape = tuple([len(patches1)])+tuple(np.shape(patches1[0]))+tuple([1]), dtype=np.float32)
    for k in range(0,len(patches1)):
        bP[k,:,:,0] = patches1[k][:,:]/255.0
    emb_1 = BigAIDmodel.get_layer("aff_desc").predict(bP)

    bP = np.zeros( shape=tuple([len(patches2)])+tuple(np.shape(patches2[0]))+tuple([1]), dtype=np.float32)
    for k in range(0,len(patches2)):
        bP[k,:,:,0] = patches2[k][:,:]/255.0
    emb_2 = BigAIDmodel.get_layer("aff_desc").predict(bP)

    ET_KP = time.time() - start_time

    desc_dim = np.shape(emb_1)[1]
    lda = CPPbridge('./build/libDA.so')
    lda.CreateMatcher(desc_dim, k = knn_num, sim_thres = MatchingThres)
    start_time = time.time()
    lda.KnnMatch(KPlist1,emb_1, KPlist2,emb_2,FastCode)    
    ET_M = time.time() - start_time

    AID_all = lda.GetAllMatches()
    AID_all = OnlyUniqueMatches(AID_all,KPlist1,KPlist2,SpatialThres=SameKPthres)

    return KPlist1, KPlist2, AID_all, pyr1, pyr2


class MethodScore(object):
    def __init__(self, MethodName='unknown'):
        self.methodname = MethodName
        self.score = 0
        self.cInliers = 0
        self.StdInliersPerImg = []
        self.InliersInImg = []
        self.av_dist_pxls = 0
        self.ratioTruePositives = 0
        self.identified_pair_names = []
        self.saved = []
        self.seen_pair_names = []
        self.seen_pair_names_rep = {}
        self.seen_data = []

    def SaveInfo(self,cmHC,HC,total, AvDist, pair_name):
        self.saved = [cmHC,HC,total, AvDist, pair_name]
    def FlushSavedInfo(self):
        self.saved = []
    def AddSavedInfo(self):
        cmHC,HC,total, AvDist, pair_name = self.saved
        self.AddInfo(cmHC,HC,total, AvDist, pair_name)

    def AddInfo(self,cmHC,HC,total, AvDist, pair_name):
        self.InliersInImg.append( len(cmHC) )
        self.seen_data.append( [len(cmHC),len(HC),len(total), AvDist, pair_name]  )
        if not (pair_name in self.seen_pair_names):
            self.seen_pair_names.append( pair_name )
            self.seen_pair_names_rep[pair_name] = 0
        self.seen_pair_names_rep[pair_name] += 1
        if len(total)>0 and len(cmHC)>0.8*len(HC):
             self.score += 1
             self.cInliers += len(cmHC)
             self.ratioTruePositives += len(cmHC)/len(total)
             self.av_dist_pxls += AvDist
             if not (pair_name in self.identified_pair_names):
                 self.identified_pair_names.append( pair_name )

    def EndOfImgTreatement(self):
        if len(self.InliersInImg)>0:
            self.StdInliersPerImg.append( np.std(self.InliersInImg) )
            self.InliersInImg = []

    def GetPerImgAnalysis(self):
        if len(self.StdInliersPerImg)>0 and self.score>0:            
            return "; C.In.STD: " + str(np.mean(self.StdInliersPerImg))
        else:
            return ""

    def GetScore(self, inStr = True):
        if inStr:
            if self.score>0:
                return "Rep.I.Pairs (I.Pairs): " + str(self.score)+ "(" + str(len(self.identified_pair_names)) + ")" + " ; Av. C.In.: " + str(self.cInliers/self.score) + "; Av. Error: " + str(self.av_dist_pxls/self.score) +"; TruePos: " +str(self.ratioTruePositives/self.score)
            else:
                return 'No success in this dataset'
        else:
            if self.score>0:
                return self.score, self.cInliers/self.score, self.av_dist_pxls/self.score, self.ratioTruePositives/self.score
            else:
                return 0, -1, -1


def CorrectMatches(matches,kplistq, kplistt, H, thres = 24):
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

ds = LoadDatasets()
lda = CPPbridge('./build/libDA.so')

def LaunchAndRecord_USAC(MS, p, kplistq, kplistt, total, iters=1):
    for i in range(iters):
        good_HC, H = lda.GeometricFilter(kplistq, p.query, kplistt, p.target, total, Filter=GeoFilter, precision=24)
        cmHC, AvDist = CorrectMatches(good_HC,kplistq, kplistt, p.Tmatrix )
        cmT, _ = CorrectMatches(total,kplistq, kplistt, p.Tmatrix )
        MS.AddInfo(cmHC, good_HC, total, AvDist, p.pair_name)
        # print("----> %s : cmT = %d, cmHC = %d(%3.2f), HC = %d, Total = %d" %(MS.methodname, len(cmT), len(cmHC), AvDist, len(good_HC), len(total)))
    MS.EndOfImgTreatement()
    
def DoGuidedMatching(matches_all, img1, img2, pyr1, pyr2, KPlist1, KPlist2, RegionGrowingIters=4, LocalGeoEsti = '8pts'):
    h1, w1 = img1.shape[:2]
    Qmap = np.zeros(shape=[w1,h1],dtype=bool)
    KPlist1 = features_deepcopy(KPlist1)
    KPlist2 = features_deepcopy(KPlist2)
    matches_all = matches_deepcopy(matches_all)
    KPlist1, KPlist2, growed_matches, growing_matches, Qmap = GrowMatches(KPlist1, pyr1, KPlist2, pyr2, Qmap, [], matches_all, img1, img2, LocalGeoEsti=LocalGeoEsti)
    for i in range(RegionGrowingIters):
        KPlist1, KPlist2, growed_matches, growing_matches, Qmap = GrowMatches(KPlist1, pyr1, KPlist2, pyr2, Qmap, growed_matches, growing_matches, img1, img2, LocalGeoEsti=LocalGeoEsti)
        if len(growing_matches)==0:
            break
    return growed_matches, KPlist1, KPlist2


RANSACiters = 100
dsvec = [0] # [0,1], [2,3] or [4,0,1]

import pickle
for i in tqdm(dsvec):    
    storepicklepath = './store_GM_'+str(i)+'.pckl'
    SAH = MethodScore(MethodName='SIFT_AffNet_HardNet')
    SAHRV = MethodScore(MethodName='SIFT_AffNet_HardNet RANSAC Vanilla')
    SAHR2 = MethodScore(MethodName='SIFT_AffNet_HardNet RANSAC 2pts')
    SAHRA = MethodScore(MethodName='SIFT_AffNet_HardNet RANSAC affine')
    SAHI = MethodScore(MethodName='Guided_SIFT_AffNet_HardNet with identity')
    SAH8 = MethodScore(MethodName='Guided_SIFT_AffNet_HardNet with 8pts')
    SAH4 = MethodScore(MethodName='Guided_SIFT_AffNet_HardNet with 4pts')
    SAHA = MethodScore(MethodName='Guided_SIFT_AffNet_HardNet with affnet')
    SA = MethodScore(MethodName='SIFT-AID')
    SARV = MethodScore(MethodName='SIFT-AID RANSAC Vanilla')
    SAR2 = MethodScore(MethodName='SIFT-AID RANSAC 2pts')
    SARA = MethodScore(MethodName='SIFT-AID RANSAC affine')
    SAI = MethodScore(MethodName='Guided_SIFT-AID with identity')
    SA8 = MethodScore(MethodName='Guided_SIFT-AID with 8pts')
    SA4 = MethodScore(MethodName='Guided_SIFT-AID with 4pts')
    SAA = MethodScore(MethodName='Guided_SIFT-AID with affnet')

    try:
        f = open(storepicklepath, 'rb')
        SAH, SAHRV, SAHR2, SAHRA, SAHI, SAH8, SAH4, SAHA, SA, SARV, SAR2, SARA, SAI, SA8, SA4, SAA = pickle.load(f)
        f.close()
        print('Loading PICKLE-STORAGE complete')
    except:
        print('Skipping PICKLE')

    StoreMSs = [SAH, SAHRV, SAHR2, SAHRA, SAHI, SAH8, SAH4, SAHA, SA, SARV, SAR2, SARA, SAI, SA8, SA4, SAA]
    for p in tqdm(ds[i].datapairs):
        if p.pair_name in SAHRV.seen_pair_names or p.pair_name in SAH.seen_pair_names:
            print('already done:',p.pair_name)
            continue
        # print(p.pair_name)
        
        # SIFT-Affnet-Hardnet
        kplistq, kplistt, total, Aq2t_list, pyr1, pyr2 = ModifiedSIFT_AffNet_HardNet(p.query,p.target)
        LaunchAndRecord_USAC(SAH, p, kplistq, kplistt, total, iters=RANSACiters)

        LaunchAndRecord_Aff_RANSAC_H(SAHRV, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 0)

        LaunchAndRecord_Aff_RANSAC_H(SAHR2, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 1)

        LaunchAndRecord_Aff_RANSAC_H(SAHRA, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 2)
        
        newtotal, newkplistq, newkplistt = DoGuidedMatching(total, p.query, p.target, pyr1, pyr2, kplistq, kplistt, LocalGeoEsti = 'identity')
        LaunchAndRecord_USAC(SAHI, p, newkplistq, newkplistt, newtotal, iters=RANSACiters)
        
        newtotal, newkplistq, newkplistt = DoGuidedMatching(total, p.query, p.target, pyr1, pyr2, kplistq, kplistt, LocalGeoEsti = '8pts')
        LaunchAndRecord_USAC(SAH8, p, newkplistq, newkplistt, newtotal, iters=RANSACiters)

        # newtotal, newkplistq, newkplistt = DoGuidedMatching(total, p.query, p.target, pyr1, pyr2, kplistq, kplistt, LocalGeoEsti = '4pts')
        # LaunchAndRecord_USAC(SAH4, p, newkplistq, newkplistt, newtotal, iters=RANSACiters)

        newtotal, newkplistq, newkplistt = DoGuidedMatching(total, p.query, p.target, pyr1, pyr2, kplistq, kplistt, LocalGeoEsti = 'affnet')
        LaunchAndRecord_USAC(SAHA, p, newkplistq, newkplistt, newtotal, iters=RANSACiters)

        # SIFT-AID
        kplistq, kplistt, total, pyr1, pyr2 = Modified_siftAID(p.query,p.target,MatchingThres = SignAlingThres, Simi = 'SignProx')
        LaunchAndRecord_USAC(SA, p, kplistq, kplistt, total, iters=RANSACiters)

        LaunchAndRecord_Aff_RANSAC_H(SARV, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 0)

        LaunchAndRecord_Aff_RANSAC_H(SAR2, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 1)

        LaunchAndRecord_Aff_RANSAC_H(SARA, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 2)

        newtotal, newkplistq, newkplistt = DoGuidedMatching(total, p.query, p.target, pyr1, pyr2, kplistq, kplistt, LocalGeoEsti = 'identity')
        LaunchAndRecord_USAC(SAI, p, newkplistq, newkplistt, newtotal, iters=RANSACiters)

        newtotal, newkplistq, newkplistt = DoGuidedMatching(total, p.query, p.target, pyr1, pyr2, kplistq, kplistt, LocalGeoEsti = '8pts')
        LaunchAndRecord_USAC(SA8, p, newkplistq, newkplistt, newtotal, iters=RANSACiters)

        # newtotal, newkplistq, newkplistt = DoGuidedMatching(total, p.query, p.target, pyr1, pyr2, kplistq, kplistt, LocalGeoEsti = '4pts')
        # LaunchAndRecord_USAC(SA4, p, newkplistq, newkplistt, newtotal, iters=RANSACiters)

        newtotal, newkplistq, newkplistt = DoGuidedMatching(total, p.query, p.target, pyr1, pyr2, kplistq, kplistt, LocalGeoEsti = 'affnet')
        LaunchAndRecord_USAC(SAA, p, newkplistq, newkplistt, newtotal, iters=RANSACiters)

        f = open(storepicklepath, 'wb')
        pickle.dump(StoreMSs, f)
        f.close()
    
    print(' ')
    print('Summary on',ds[i].name)
    for MS in StoreMSs:
        print('---->',MS.methodname.center(40,' '), ': ',MS.GetScore(), MS.GetPerImgAnalysis())
    