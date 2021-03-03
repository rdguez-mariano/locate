import cv2
import sys
sys.path.append(".")
from library import *
from acc_test_library import *
from libLocalDesc import *
from tqdm import tqdm

CosProxThres = 0.4
SignAlingThres = 4000
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



def ModifiedSIFT_HardNet(img1,img2, MatchingThres = 0.8, knn_num = 2):
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
    emb_1, bP_Alist1 = HardNet_describe(bP)

    bP = np.zeros( shape=tuple([len(patches2)])+tuple(np.shape(patches2[0]))+tuple([1]), dtype=np.float32)
    for k in range(0,len(patches2)):
        bP[k,:,:,0] = patches2[k][:,:]
    emb_2, bP_Alist2 = HardNet_describe(bP)
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
    
    from AffRANSAC import Aff_RANSAC_H
    Aq2t_locate = get_Aq2t(A_list1, patches1, A_list2, patches2, sift_all, method='locate')
    Aq2t_affnet = get_Aq2t(A_list1, patches1, A_list2, patches2, sift_all, method='affnet')
    Aq2t_identity = get_Aq2t(A_list1, patches1, A_list2, patches2, sift_all, method='simple')

    return KPlist1, KPlist2, sift_all, Aq2t_locate, Aq2t_affnet, Aq2t_identity


def Modified_siftAID(img1,img2, MatchingThres = math.inf, Simi='SignProx', knn_num = 1, descRadius=math.inf):
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

    bP = np.zeros( shape = tuple([len(patches1)])+tuple(np.shape(patches1[0]))+tuple([1]), dtype=np.float32)
    for k in range(0,len(patches1)):
        bP[k,:,:,0] = patches1[k][:,:]/255.0
    
    global graph
    global tfsession
    with graph.as_default(): 
        with tfsession.as_default():       
            emb_1 = BigAIDmodel.get_layer("aff_desc").predict(bP)
            emb_1 = CreateSubDesc(emb_1, coef=1.0, NewDescRadius=descRadius)

    bP = np.zeros( shape=tuple([len(patches2)])+tuple(np.shape(patches2[0]))+tuple([1]), dtype=np.float32)
    for k in range(0,len(patches2)):
        bP[k,:,:,0] = patches2[k][:,:]/255.0
    with graph.as_default(): 
        with tfsession.as_default():
            emb_2 = BigAIDmodel.get_layer("aff_desc").predict(bP)
            emb_2 = CreateSubDesc(emb_2, coef=-1.0, NewDescRadius=descRadius)
    
    ET_KP = time.time() - start_time

    desc_dim = np.shape(emb_1)[1]
    lda = CPPbridge('./build/libDA.so')
    lda.CreateMatcher(desc_dim, k = knn_num, sim_thres = MatchingThres)
    start_time = time.time()
    lda.KnnMatch(KPlist1,emb_1, KPlist2,emb_2,FastCode)    
    ET_M = time.time() - start_time

    AID_all = lda.GetAllMatches()
    AID_all = OnlyUniqueMatches(AID_all,KPlist1,KPlist2,SpatialThres=SameKPthres)

    from AffRANSAC import Aff_RANSAC_H
    Aq2t_locate = get_Aq2t(A_list1, patches1, A_list2, patches2, AID_all, method='locate')
    Aq2t_affnet = get_Aq2t(A_list1, patches1, A_list2, patches2, AID_all, method='affnet')
    Aq2t_identity = get_Aq2t(A_list1, patches1, A_list2, patches2, AID_all, method='simple')

    return KPlist1, KPlist2, AID_all, Aq2t_locate, Aq2t_affnet, Aq2t_identity


def Modified_HessAffAID(img1,img2, Ndesc=500, MatchingThres = math.inf, Simi='SignProx', knn_num = 1, descRadius=math.inf):
    if Simi=='CosProx':
        FastCode = 0
    elif Simi=='SignProx':
        FastCode = 1
    elif Simi=='SignProxKNN':
        FastCode = 2
    else:
        print('Wrong similarity choice for HessAffSIFT !!!')
        exit()

    # find the keypoints with SIFT
    start_time = time.time()
    KPlist1, patches1, Alist1, responses1 = HessAff_Detect(img1, PatchSize=60, Nfeatures=Ndesc)
    KPlist2, patches2, Alist2, responses2 = HessAff_Detect(img2, PatchSize=60, Nfeatures=Ndesc)    
    
    bP = np.zeros( shape = tuple([patches1.shape[0], patches1.shape[2], patches1.shape[3], 1]), dtype=np.float32)
    for k in range(0,len(patches1)):
        bP[k,:,:,0] = patches1[k,0,:,:]/255.0
    global graph
    global tfsession
    with graph.as_default(): 
        with tfsession.as_default():       
            emb_1 = BigAIDmodel.get_layer("aff_desc").predict(bP)
            emb_1 = CreateSubDesc(emb_1, coef=1.0, NewDescRadius=descRadius)

    bP = np.zeros( shape = tuple([patches2.shape[0], patches2.shape[2], patches2.shape[3], 1]), dtype=np.float32)
    for k in range(0,len(patches2)):
        bP[k,:,:,0] = patches2[k,0,:,:]/255.0
    with graph.as_default(): 
        with tfsession.as_default():    
            emb_2 = BigAIDmodel.get_layer("aff_desc").predict(bP)
            emb_2 = CreateSubDesc(emb_2, coef=-1.0, NewDescRadius=descRadius)
    
    ET_KP = time.time() - start_time

    desc_dim = np.shape(emb_1)[1]
    lda = CPPbridge('./build/libDA.so')
    lda.CreateMatcher(desc_dim, k = knn_num, sim_thres = MatchingThres)
    start_time = time.time()
    lda.KnnMatch(KPlist1,emb_1, KPlist2,emb_2,FastCode)    
    ET_M = time.time() - start_time

    AID_all = []
    
    from AffRANSAC import Aff_RANSAC_H
    AID_all = lda.GetAllMatches()
    AID_all = OnlyUniqueMatches(AID_all,KPlist1,KPlist2,SpatialThres=5)
    Alist1 = [cv2.invertAffineTransform(A) for A in Alist1]
    Alist2 = [cv2.invertAffineTransform(A) for A in Alist2]
    
    Aq2t_locate = get_Aq2t(Alist1, patches1[:,0,:,:], Alist2, patches2[:,0,:,:], AID_all, method='locate', noTranslation=True)
    Aq2t_affnet = get_Aq2t(Alist1, patches1[:,0,:,:], Alist2, patches2[:,0,:,:], AID_all, method='affnet', noTranslation=True)
    Aq2t_identity = get_Aq2t(Alist1, patches1[:,0,:,:], Alist2, patches2[:,0,:,:], AID_all, method='simple', noTranslation=True)
    
    return KPlist1, KPlist2, AID_all, Aq2t_locate, Aq2t_affnet, Aq2t_identity


def Modified_HessAff_HardNet(img1,img2, MatchingThres = 0.8, Ndesc=500, HessAffNet=False):
    start_time = time.time()
    if HessAffNet:
        KPlist1, Patches1, descriptors1, Alist1, Score1 = HessAffNetHardNet_DetectAndDescribe(img1, Nfeatures=Ndesc)
        KPlist2, Patches2, descriptors2, Alist2, Score2 = HessAffNetHardNet_DetectAndDescribe(img2, Nfeatures=Ndesc)
    else:
        KPlist1, Patches1, descriptors1, Alist1, Score1 = HessAffNetHardNet_DetectAndDescribe(img1, Nfeatures=Ndesc, useAffnet=None)
        KPlist2, Patches2, descriptors2, Alist2, Score2 = HessAffNetHardNet_DetectAndDescribe(img2, Nfeatures=Ndesc, useAffnet=None)
    ET_KP = time.time() - start_time

    #Bruteforce matching with SNN threshold    
    start_time = time.time()
    tent_matches_in_1, tent_matches_in_2 = BruteForce4HardNet(descriptors1,descriptors2, SNN_threshold=MatchingThres)
    ET_M = time.time() - start_time

    sift_all = OnlyUniqueMatches( [cv2.DMatch(i, j, 1.0) for i,j in zip(tent_matches_in_1,tent_matches_in_2)], KPlist1, KPlist2 )

    
    from AffRANSAC import Aff_RANSAC_H
    Alist1 = [cv2.invertAffineTransform(A) for A in Alist1]
    Alist2 = [cv2.invertAffineTransform(A) for A in Alist2]
    Aq2t_locate = get_Aq2t(Alist1, Patches1[:,0,:,:], Alist2, Patches2[:,0,:,:], sift_all, method='locate', noTranslation=True)
    Aq2t_affnet = get_Aq2t(Alist1, Patches1[:,0,:,:], Alist2, Patches2[:,0,:,:], sift_all, method='affnet', noTranslation=True)
    Aq2t_identity = get_Aq2t(Alist1, Patches1[:,0,:,:], Alist2, Patches2[:,0,:,:], sift_all, method='simple', noTranslation=True)
    
    return KPlist1, KPlist2, sift_all, Aq2t_locate, Aq2t_affnet, Aq2t_identity


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

def LaunchAndRecord_USAC(MS, p, kplistq, kplistt, total, iters=1, GeoFilter = 'USAC_H'):
    for i in range(iters):
        good_HC, H = lda.GeometricFilter(kplistq, p.query, kplistt, p.target, total, Filter=GeoFilter, precision=24)
        cmHC, AvDist = CorrectMatches(good_HC,kplistq, kplistt, p.Tmatrix )
        cmT, _ = CorrectMatches(total,kplistq, kplistt, p.Tmatrix )
        MS.AddInfo(cmHC, good_HC, total, AvDist, p.pair_name)
        # print("----> %s : cmT = %d, cmHC = %d(%3.2f), HC = %d, Total = %d" %(MS.methodname, len(cmT), len(cmHC), AvDist, len(good_HC), len(total)))
    MS.EndOfImgTreatement()


RANSACiters = 100
dsvec = [0] # [0,1], [2,3] or [4,0,1]

import pickle
for i in tqdm(dsvec):    
    storepicklepath = './store_GM_'+str(i)+'.pckl'

    SHR = MethodScore(MethodName='SIFT-HardNet RANSAC')
    SHR2loc = MethodScore(MethodName='SIFT-HardNet RANSAC 2pts locate')
    SHRAloc = MethodScore(MethodName='SIFT-HardNet RANSAC affine locate')
    SHR2aff = MethodScore(MethodName='SIFT-HardNet RANSAC 2pts affnet')
    SHRAaff = MethodScore(MethodName='SIFT-HardNet RANSAC affine affnet')
    SHR2id = MethodScore(MethodName='SIFT-HardNet RANSAC 2pts identity')
    SHRAid = MethodScore(MethodName='SIFT-HardNet RANSAC affine identity')
    SHRU = MethodScore(MethodName='SIFT-HardNet RANSAC USAC')
    SHRO = MethodScore(MethodName='SIFT-HardNet RANSAC ORSA')

    SAR = MethodScore(MethodName='SIFT-AID RANSAC')
    SAR2loc = MethodScore(MethodName='SIFT-AID RANSAC 2pts locate')
    SARAloc = MethodScore(MethodName='SIFT-AID RANSAC affine locate')
    SAR2aff = MethodScore(MethodName='SIFT-AID RANSAC 2pts affnet')
    SARAaff = MethodScore(MethodName='SIFT-AID RANSAC affine affnet')
    SAR2id = MethodScore(MethodName='SIFT-AID RANSAC 2pts identity')
    SARAid = MethodScore(MethodName='SIFT-AID RANSAC affine identity')
    SARU = MethodScore(MethodName='SIFT-AID RANSAC USAC')
    SARO = MethodScore(MethodName='SIFT-AID RANSAC ORSA')

    HHR = MethodScore(MethodName='HessAffine-HardNet RANSAC')
    HHR2loc = MethodScore(MethodName='HessAffine-HardNet RANSAC 2pts locate')
    HHRAloc = MethodScore(MethodName='HessAffine-HardNet RANSAC affine locate')
    HHR2aff = MethodScore(MethodName='HessAffine-HardNet RANSAC 2pts affnet')
    HHRAaff = MethodScore(MethodName='HessAffine-HardNet RANSAC affine affnet')
    HHR2id = MethodScore(MethodName='HessAffine-HardNet RANSAC 2pts identity')
    HHRAid = MethodScore(MethodName='HessAffine-HardNet RANSAC affine identity')
    HHRU = MethodScore(MethodName='HessAffine-HardNet RANSAC USAC')
    HHRO = MethodScore(MethodName='HessAffine-HardNet RANSAC ORSA')

    HAR = MethodScore(MethodName='HessAffine-AID RANSAC')
    HAR2loc = MethodScore(MethodName='HessAffine-AID RANSAC 2pts locate')
    HARAloc = MethodScore(MethodName='HessAffine-AID RANSAC affine locate')
    HAR2aff = MethodScore(MethodName='HessAffine-AID RANSAC 2pts affnet')
    HARAaff = MethodScore(MethodName='HessAffine-AID RANSAC affine affnet')
    HAR2id = MethodScore(MethodName='HessAffine-AID RANSAC 2pts identity')
    HARAid = MethodScore(MethodName='HessAffine-AID RANSAC affine identity')
    HARU = MethodScore(MethodName='HessAffine-AID RANSAC USAC')
    HARO = MethodScore(MethodName='HessAffine-AID RANSAC ORSA')


    try:
        f = open(storepicklepath, 'rb')
        SHR , SHR2loc , SHRAloc , SHR2aff , SHRAaff , SHR2id , SHRAid , SHRU , SHRO , SAR , SAR2loc , SARAloc , SAR2aff , SARAaff , SAR2id , SARAid , SARU , SARO , HHR , HHR2loc , HHRAloc , HHR2aff , HHRAaff , HHR2id , HHRAid , HHRU , HHRO , HAR , HAR2loc , HARAloc , HAR2aff , HARAaff , HAR2id , HARAid , HARU , HARO = pickle.load(f)
        f.close()
        print('Loading PICKLE-STORAGE complete')
    except:
        print('Skipping PICKLE')

    StoreMSs = [SHR , SHR2loc , SHRAloc , SHR2aff , SHRAaff , SHR2id , SHRAid , SHRU , SHRO , SAR , SAR2loc , SARAloc , SAR2aff , SARAaff , SAR2id , SARAid , SARU , SARO , HHR , HHR2loc , HHRAloc , HHR2aff , HHRAaff , HHR2id , HHRAid , HHRU , HHRO , HAR , HAR2loc , HARAloc , HAR2aff , HARAaff , HAR2id , HARAid , HARU , HARO]
    for p in tqdm(ds[i].datapairs):
        if p.pair_name in SHR.seen_pair_names:
            print('already done:',p.pair_name)
            continue
        # print(p.pair_name)

        ### SIFT + HARDNET
        kplistq, kplistt, total, Aq2t_locate, Aq2t_affnet, Aq2t_identity = ModifiedSIFT_HardNet(p.query, p.target, MatchingThres = 0.8)

        LaunchAndRecord_Aff_RANSAC_H(SHR, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 0)

        LaunchAndRecord_Aff_RANSAC_H(SHR2loc, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 1, Aq2t=Aq2t_locate)
        LaunchAndRecord_Aff_RANSAC_H(SHRAloc, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 2, Aq2t=Aq2t_locate)
        
        LaunchAndRecord_Aff_RANSAC_H(SHR2aff, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 1, Aq2t=Aq2t_affnet)
        LaunchAndRecord_Aff_RANSAC_H(SHRAaff, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 2, Aq2t=Aq2t_affnet)

        LaunchAndRecord_Aff_RANSAC_H(SHR2id, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 1, Aq2t=Aq2t_identity)
        LaunchAndRecord_Aff_RANSAC_H(SHRAid, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 2, Aq2t=Aq2t_identity)

        LaunchAndRecord_USAC(SHRU, p, kplistq, kplistt, total, iters=RANSACiters, GeoFilter='USAC_H')
        LaunchAndRecord_USAC(SHRO, p, kplistq, kplistt, total, iters=RANSACiters, GeoFilter='ORSA_H')


        ### SIFT + AID
        kplistq, kplistt, total, Aq2t_locate, Aq2t_affnet, Aq2t_identity = Modified_siftAID(p.query,p.target,MatchingThres = SignAlingThres, Simi = 'SignProx')

        LaunchAndRecord_Aff_RANSAC_H(SAR, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 0)

        LaunchAndRecord_Aff_RANSAC_H(SAR2loc, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 1, Aq2t=Aq2t_locate)
        LaunchAndRecord_Aff_RANSAC_H(SARAloc, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 2, Aq2t=Aq2t_locate)
        
        LaunchAndRecord_Aff_RANSAC_H(SAR2aff, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 1, Aq2t=Aq2t_affnet)
        LaunchAndRecord_Aff_RANSAC_H(SARAaff, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 2, Aq2t=Aq2t_affnet)

        LaunchAndRecord_Aff_RANSAC_H(SAR2id, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 1, Aq2t=Aq2t_identity)
        LaunchAndRecord_Aff_RANSAC_H(SARAid, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 2, Aq2t=Aq2t_identity)

        LaunchAndRecord_USAC(SARU, p, kplistq, kplistt, total, iters=RANSACiters, GeoFilter='USAC_H')
        LaunchAndRecord_USAC(SARO, p, kplistq, kplistt, total, iters=RANSACiters, GeoFilter='ORSA_H')


        ### HESSAFFINE + HARDNET
        kplistq, kplistt, total, Aq2t_locate, Aq2t_affnet, Aq2t_identity = Modified_HessAff_HardNet(p.query, p.target, MatchingThres = 0.8, HessAffNet=False)

        LaunchAndRecord_Aff_RANSAC_H(HHR, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 0)

        LaunchAndRecord_Aff_RANSAC_H(HHR2loc, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 1, Aq2t=Aq2t_locate)
        LaunchAndRecord_Aff_RANSAC_H(HHRAloc, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 2, Aq2t=Aq2t_locate)
        
        LaunchAndRecord_Aff_RANSAC_H(HHR2aff, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 1, Aq2t=Aq2t_affnet)
        LaunchAndRecord_Aff_RANSAC_H(HHRAaff, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 2, Aq2t=Aq2t_affnet)

        LaunchAndRecord_Aff_RANSAC_H(HHR2id, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 1, Aq2t=Aq2t_identity)
        LaunchAndRecord_Aff_RANSAC_H(HHRAid, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 2, Aq2t=Aq2t_identity)

        LaunchAndRecord_USAC(HHRU, p, kplistq, kplistt, total, iters=RANSACiters, GeoFilter='USAC_H')
        LaunchAndRecord_USAC(HHRO, p, kplistq, kplistt, total, iters=RANSACiters, GeoFilter='ORSA_H')


        ### HESSAFFINE + AID
        kplistq, kplistt, total, Aq2t_locate, Aq2t_affnet, Aq2t_identity = Modified_HessAffAID(p.query,p.target, MatchingThres = SignAlingThres, Simi = 'SignProx')

        LaunchAndRecord_Aff_RANSAC_H(HAR, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 0)

        LaunchAndRecord_Aff_RANSAC_H(HAR2loc, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 1, Aq2t=Aq2t_locate)
        LaunchAndRecord_Aff_RANSAC_H(HARAloc, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 2, Aq2t=Aq2t_locate)
        
        LaunchAndRecord_Aff_RANSAC_H(HAR2aff, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 1, Aq2t=Aq2t_affnet)
        LaunchAndRecord_Aff_RANSAC_H(HARAaff, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 2, Aq2t=Aq2t_affnet)

        LaunchAndRecord_Aff_RANSAC_H(HAR2id, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 1, Aq2t=Aq2t_identity)
        LaunchAndRecord_Aff_RANSAC_H(HARAid, p, kplistq, kplistt, total, BigIters=RANSACiters, AffInfo = 2, Aq2t=Aq2t_identity)

        LaunchAndRecord_USAC(HARU, p, kplistq, kplistt, total, iters=RANSACiters, GeoFilter='USAC_H')
        LaunchAndRecord_USAC(HARO, p, kplistq, kplistt, total, iters=RANSACiters, GeoFilter='ORSA_H')

        f = open(storepicklepath, 'wb')
        pickle.dump(StoreMSs, f)
        f.close()
    
    print(' ')
    print('Summary on',ds[i].name)
    for MS in StoreMSs:
        print('---->',MS.methodname.center(50,' '), ': ',MS.GetScore(), MS.GetPerImgAnalysis())
    