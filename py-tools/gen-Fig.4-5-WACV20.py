import cv2
import sys
sys.path.append(".")
from libLocalDesc import *
from acc_test_library import *
from tqdm import tqdm
from matplotlib import pyplot as plt
import csv
import glob, os

plt.switch_backend('agg')

CosProxThres = 0.4
SignAlingThres = 4000
GeoFilter = 'USAC_H'


###  Define models to treat
geoestiMODEL_NAME = 'DA_Pts_dropout'
geoestiweights2load = 'model-data/model.'+geoestiMODEL_NAME+'_L1_75.hdf5'
geoesti_dropout = create_model(tuple([60,60,2]), tuple([16]), model_name = geoestiMODEL_NAME, Norm='L1', resume = True, ResumeFile = geoestiweights2load).get_layer("GeometricEstimator")


def WriteImgKeys(img, keys, pathname, Flag=2):
        colors=( (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255) )
        if len(np.shape(img))==2:
            patch = cv2.cvtColor( img.astype(np.uint8), cv2.COLOR_GRAY2RGB )
        else:
            patch = img.astype(np.uint8)
        if len(keys)>len(colors):
            patch=cv2.drawKeypoints(patch,keys,patch, flags=Flag)
        else:
            for n in range(np.min([len(colors),len(keys)])):
                patch=cv2.drawKeypoints(patch,keys[n:n+1],patch, color=colors[n] ,flags=Flag)
        cv2.imwrite('temp/'+pathname,patch)


def ComputeModelData(model, inputs, WasNetAffine = True):
    ''' This function is a modified version of DA_ComputeAccuracy in acc_test_library.py
    '''
    GA = GenAffine("", DryRun=True)
    h, w = 60, 60
    SquarePatch = SquareOrderedPts(h,w,CV=False)
    good = 0
    diffs_GT = []
    totalkps = 0
    pid = 0
    for input_i in inputs:        
        asift_KPlist1, patches1, GT_Avec_list, asift_KPlist2, patches2 = input_i
        assert len(asift_KPlist1)==len(patches1)==len(GT_Avec_list)==len(asift_KPlist2)==len(patches2)

        patchshape = np.shape(patches1[0])
        bPshape = tuple([1]) + tuple( patchshape ) + tuple([2])
        bP = np.zeros(shape=bPshape, dtype = np.float32)
        Identity = np.float32([[1, 0, 0], [0, 1, 0]])
        Ivec = affine_decomp(Identity) # [1,0,1,0,0,0]
        
        totalkps += len(asift_KPlist1)
        
        for k in range(0,len(asift_KPlist1)):
            bP[0,:,:,:] = np.dstack((patches1[k]/GA.imgdivfactor, patches2[k]/GA.imgdivfactor))

            # Estimated
            Avec = model.predict(bP)            
            if WasNetAffine:
                Avec = Avec[0]*GA.Avec_factor + GA.Avec_tras
                A = np.reshape( affine_decomp2affine( Avec[0:6] ), (2,3) ) # transforms p2 into p1 coordinates
            else:
                A = GA.AffineFromNormalizedVector( Avec[0] ) # transforms p2 into p1 coordinates
                vecE = Avec[0]
                Avec = affine_decomp(A,doAssert=False)

            Ai = cv2.invertAffineTransform(A)
            Aivec = affine_decomp(Ai,doAssert=False)

            # Groundtruth
            GTAvec = GT_Avec_list[k] # transforms p1 into p2 coordinates
            GTA = np.reshape( affine_decomp2affine( GTAvec ), (2,3) )
            GTAi = cv2.invertAffineTransform( GTA )
            GTAivec = affine_decomp( GTAi )

            diffs_GT.append( np.array(GTAivec) - np.array(Avec) )
            diffs_GT.append( np.array(GTAvec) - np.array(Aivec) )

            if transition_tilt(Avec,GTAivec)<=transition_tilt(Ivec,GTAivec):
                good+=1
            
            # pid += 1
            # Affdiff = [ GTAivec[0]/Avec[0] if GTAivec[0]>Avec[0] else Avec[0]/GTAivec[0], 
            #                 AngleDiff(GTAivec[1],Avec[1],InRad=True), 
            #                 GTAivec[2]/Avec[2] if GTAivec[2]>Avec[2] else Avec[2]/GTAivec[2] , 
            #                 AngleDiff(GTAivec[3],Avec[3],InRad=True) ]
            # if (Affdiff>np.array( [1.1, 0.0, 1.1, 0.0] )).all() and (Affdiff<np.array( [2.0, np.pi/20, 2.0, np.pi/20] )).all():
            #     WriteImgKeys(patches1[k], [], 'p1/'+str(pid)+'.png' )
            #     WriteImgKeys(patches2[k], [], 'p2/'+str(pid)+'.png' )
            #     GenAffineMapsImage(GTAivec, vecE, 'p1init/'+str(pid)+'.png', GA, SquarePatch)

    return np.array(diffs_GT), np.float(good)/totalkps

import sys
sys.path.append("hesaffnet")
from hesaffnet import *
def ComputeAffNetData( inputs):
    ''' This function is a modified version of DA_ComputeAccuracy in acc_test_library.py
    '''
    GA = GenAffine("", DryRun=True)
    h, w = 60, 60
    SquarePatch = SquareOrderedPts(h,w,CV=False)
    good = 0
    diffs_GT = []
    totalkps = 0
    pid = 0
    for input_i in inputs:        
        asift_KPlist1, patches1, GT_Avec_list, asift_KPlist2, patches2 = input_i
        assert len(asift_KPlist1)==len(patches1)==len(GT_Avec_list)==len(asift_KPlist2)==len(patches2)

        patchshape = np.shape(patches1[0])
        bPshape = tuple([1]) + tuple( patchshape ) + tuple([2])
        bP = np.zeros(shape=bPshape, dtype = np.float32)
        Identity = np.float32([[1, 0, 0], [0, 1, 0]])
        Ivec = affine_decomp(Identity) # [1,0,1,0,0,0]
        
        totalkps += len(asift_KPlist1)
        
        emb_1, bP_Alist1 = AffNetHardNet_describe(np.expand_dims(np.array(patches1),axis=3))
        emb_2, bP_Alist2 = AffNetHardNet_describe(np.expand_dims(np.array(patches2),axis=3))

        for k in range(0,len(asift_KPlist1)):
            Ai = ComposeAffineMaps( bP_Alist2[k], cv2.invertAffineTransform(bP_Alist1[k]) )
            A = cv2.invertAffineTransform(Ai) # transforms p2 into p1 coordinates
            
            Avec = affine_decomp(A,doAssert=False)
            Aivec = affine_decomp(Ai,doAssert=False)

            # Groundtruth
            GTAvec = GT_Avec_list[k] # transforms p1 into p2 coordinates
            GTA = np.reshape( affine_decomp2affine( GTAvec ), (2,3) )
            GTAi = cv2.invertAffineTransform( GTA )
            GTAivec = affine_decomp( GTAi )

            diffs_GT.append( np.array(GTAivec) - np.array(Avec) )
            diffs_GT.append( np.array(GTAvec) - np.array(Aivec) )

            if transition_tilt(Avec,GTAivec)<=transition_tilt(Ivec,GTAivec):
                good+=1
            
            # pid += 1
            # Affdiff = [ GTAivec[0]/Avec[0] if GTAivec[0]>Avec[0] else Avec[0]/GTAivec[0], 
            #                 AngleDiff(GTAivec[1],Avec[1],InRad=True), 
            #                 GTAivec[2]/Avec[2] if GTAivec[2]>Avec[2] else Avec[2]/GTAivec[2] , 
            #                 AngleDiff(GTAivec[3],Avec[3],InRad=True) ]
            # if (Affdiff>np.array( [1.1, 0.0, 1.1, 0.0] )).all() and (Affdiff<np.array( [2.0, np.pi/20, 2.0, np.pi/20] )).all():
            #     WriteImgKeys(patches1[k], [], 'p1/'+str(pid)+'.png' )
            #     WriteImgKeys(patches2[k], [], 'p2/'+str(pid)+'.png' )
            #     GenAffineMapsImage(GTAivec, vecE, 'p1init/'+str(pid)+'.png', GA, SquarePatch)

    return np.array(diffs_GT), np.float(good)/totalkps


def ComputeIdentityData(inputs):
    ''' This function is a modified version of DA_ComputeAccuracy in acc_test_library.py
    '''
    good = 0
    diffs_GT = []
    totalkps = 0
    pid = 0
    for input_i in inputs:        
        asift_KPlist1, patches1, GT_Avec_list, asift_KPlist2, patches2 = input_i
        assert len(asift_KPlist1)==len(patches1)==len(GT_Avec_list)==len(asift_KPlist2)==len(patches2)
   
        Identity = np.float32([[1, 0, 0], [0, 1, 0]])
        Ivec = affine_decomp(Identity) # [1,0,1,0,0,0]
        
        totalkps += len(asift_KPlist1)

        for k in range(0,len(asift_KPlist1)):
            Ai = Identity
            A = Identity
            
            Avec = Ivec
            Aivec = Ivec

            # Groundtruth
            GTAvec = GT_Avec_list[k] # transforms p1 into p2 coordinates
            GTA = np.reshape( affine_decomp2affine( GTAvec ), (2,3) )
            GTAi = cv2.invertAffineTransform( GTA )
            GTAivec = affine_decomp( GTAi )

            diffs_GT.append( np.array(GTAivec) - np.array(Avec) )
            diffs_GT.append( np.array(GTAvec) - np.array(Aivec) )

            if transition_tilt(Avec,GTAivec)<=transition_tilt(Ivec,GTAivec):
                good+=1
            
            # pid += 1
            # Affdiff = [ GTAivec[0]/Avec[0] if GTAivec[0]>Avec[0] else Avec[0]/GTAivec[0], 
            #                 AngleDiff(GTAivec[1],Avec[1],InRad=True), 
            #                 GTAivec[2]/Avec[2] if GTAivec[2]>Avec[2] else Avec[2]/GTAivec[2] , 
            #                 AngleDiff(GTAivec[3],Avec[3],InRad=True) ]
            # if (Affdiff>np.array( [1.1, 0.0, 1.1, 0.0] )).all() and (Affdiff<np.array( [2.0, np.pi/20, 2.0, np.pi/20] )).all():
            #     WriteImgKeys(patches1[k], [], 'p1/'+str(pid)+'.png' )
            #     WriteImgKeys(patches2[k], [], 'p2/'+str(pid)+'.png' )
            #     GenAffineMapsImage(GTAivec, vecE, 'p1init/'+str(pid)+'.png', GA, SquarePatch)

    return np.array(diffs_GT), np.float(good)/totalkps


def WriteHistos2Disk(info_all, hist_type = 'bar'):
    column_names = ['zoom', 'phi2', 'tilt', 'phi1', 'x-coor', 'y-coor' ]
    xlims = [(-1.5,1.5), (-3.14,3.14), (-2,2) , (-3.14,3.14), (-75,75), (-75,75)]
    for i in range(len(column_names)):
        plt.figure(1,figsize=(5,7))
        score_str = ''
        for info in info_all:
            diffs, score, name = info
            score_str += name + ' : ' +"%.2f"%(np.mean(diffs[:,i].ravel())) + '; '
            _, bpos, _ = plt.hist(diffs[:,i].ravel(), bins='auto', density=True, alpha=0.5, histtype=hist_type, label=name)
        plt.legend(loc='upper left',fontsize=16)#'upper right')
        plt.title(column_names[i]+' ('+ str(np.shape(diffs)[0])+' Patches) \n ['+score_str+']')
        # plt.show()
        plt.xlim(xlims[i])
        plt.gca().tick_params(labelsize=13)
        plt.savefig('./temp/Histo_'+column_names[i]+'.png', format='png', dpi=300)
        plt.close(1)


ConstrastSimu = False
def ProcessData(GA, stacked_patches, groundtruth_pts,sigma = 0.0):
    if ConstrastSimu:
        channels = np.int32(np.shape(stacked_patches)[2]/2)
        val1 = random.uniform(1/3, 3)
        val2 = random.uniform(1/3, 3)
        for i in range(channels):
            stacked_patches[:,:,i] = np.power(stacked_patches[:,:,i],val1)
            stacked_patches[:,:,channels+i] = np.power(stacked_patches[:,:,channels+i],val2)
        if sigma>0.0:
            gaussian = np.random.normal(0.0,sigma,(stacked_patches.shape)).astype(np.float32)
            stacked_patches = stacked_patches + gaussian
            stacked_patches[stacked_patches<0] = 0.0
            stacked_patches[stacked_patches>1] = 1.0
    return stacked_patches, groundtruth_pts #if ConstrastSimu==False -> Identity


def GenAffineMapsImage(AvecGT,vecE, pathname, GA, SquarePatch, inSquare=True):
    vecGT = GA.Avec2Nvec( (AvecGT-GA.Avec_tras)/GA.Avec_factor )
    # vecE = GA.Avec2Nvec( (AvecE-GA.Avec_tras)/GA.Avec_factor )
    fig = plt.figure(1, figsize=(7,7))
    spn = GA.NormalizeVector( Pts2Flatten(SquarePatch) )
    plt.plot(close_per(spn[0:8:2]),close_per(spn[1:8:2]),':k')
    
    plt.plot(close_per(vecE[0:8:2]),close_per(vecE[1:8:2]),'-g')
    plt.plot(close_per(vecE[8:16:2]),close_per(vecE[9:16:2]),'--g')
    A = GA.AffineFromNormalizedVector(vecE)
    vecE[0:8] = GA.NormalizeVector( Pts2Flatten(AffineArrayCoor(SquarePatch,A)) )
    
    plt.plot(close_per(vecGT[0:8:2]),close_per(vecGT[1:8:2]),'-b')
    plt.plot(close_per(vecGT[8:16:2]),close_per(vecGT[9:16:2]),'--b')
    plt.plot(close_per(vecE[0:8:2]),close_per(vecE[1:8:2]),'-r')

    vecE[8:16] = GA.NormalizeVector( Pts2Flatten(AffineArrayCoor(SquarePatch,cv2.invertAffineTransform(A))) )
    plt.plot(close_per(vecE[8:16:2]),close_per(vecE[9:16:2]),'--r')
    
    if inSquare:
        plt.axis([0, 1, 0, 1])
    plt.title("Blue - GroundTruth / Red - Affine / Green - Homography")
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.savefig('temp/'+pathname, format='png', dpi=300)
    plt.close(fig)


def affine_generator(GA, batch_num=32):
    while True:
        stacked_patches, groundtruth_pts = [], []     
        stacked_patches, groundtruth_pts = GA.Fast_gen_affine_patches()
        stacked_patches, groundtruth_pts = ProcessData(GA, stacked_patches, groundtruth_pts)
        vgg_input_shape = np.shape(stacked_patches)
        vgg_output_shape = np.shape(groundtruth_pts)
        bPshape = tuple([batch_num]) + tuple(vgg_input_shape)
        bGTshape = tuple([batch_num]) + tuple(vgg_output_shape)
        bP = np.zeros(shape=bPshape, dtype = np.float32)
        bGT = np.zeros(shape=bGTshape, dtype = np.float32)

        bP[0,:,:,:] = stacked_patches
        bGT[0,:] = groundtruth_pts

        for i in range(1,batch_num):
            stacked_patches, groundtruth_pts = GA.Fast_gen_affine_patches()
            stacked_patches, groundtruth_pts = ProcessData(GA, stacked_patches, groundtruth_pts)
            bP[i,:,:,:] = stacked_patches
            bGT[i,:] = groundtruth_pts
        yield [bP,bGT], None

def DrawinP2(p1,p2, A, Type= 2):
    if Type==1:
        mask = np.zeros((60, 60), np.uint8)
        mask[:] = 2.0
        patch_1_in_2 = cv2.warpAffine(p1*255, A, (60, 60), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        mask2 = cv2.warpAffine(mask, A, (60, 60), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        mask2[mask2<2] = 1.0
        img = (0.5*p2*255+patch_1_in_2)/mask2        
    else:
        img = np.zeros((60, 60,3), np.uint8)
        img[:,:,0] = cv2.warpAffine(p1*255, A, (60, 60), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        img[:,:,1] = p2*255

    corners = AffineKPcoor(SquareOrderedPts(61,61,CV=True), A)
    pts = np.array([np.array(kp.pt) for kp in corners], np.int32)
    pts = pts.reshape((-1,1,2))
    if len(np.shape(img))==2:
        img = cv2.cvtColor( img.astype(np.uint8), cv2.COLOR_GRAY2RGB )
    return cv2.polylines(img,[pts],True,(0,0,255),thickness=1)    

def ComputeModelDataOnSimus(model, inputs, folder, WasNetAffine = True):
    bP, bGT = inputs
    h, w = 60, 60
    mask = np.zeros((h, w), np.uint8)
    mask[:] = 2.0
    SP = SquareOrderedPts(h,w,CV=False)
    mout = model.predict(bP)
    pid = 0
    GA = GenAffine("", DryRun=True)
    for i in range(0,np.shape(bGT)[0]):
        GTA = GA.AffineFromNormalizedVector(bGT[i,:]) # transforms p2 into p1 coordinates
        GTAvec = affine_decomp(GTA)
        GTAi = cv2.invertAffineTransform( GTA )
        GTAivec = affine_decomp( GTAi )
        
        Avec = mout[i,:]            
        if WasNetAffine:
            Avec = Avec*GA.Avec_factor + GA.Avec_tras
            A = np.reshape( affine_decomp2affine( Avec[0:6] ), (2,3) ) # transforms p2 into p1 coordinates
        else:
            A = GA.AffineFromNormalizedVector(Avec) # transforms p2 into p1 coordinates
            vecE = Avec
            Avec = affine_decomp(A,doAssert=False)

        Ai = cv2.invertAffineTransform(A)
        Aivec = affine_decomp(Ai,doAssert=False)
        
        WriteImgKeys(bP[i,:,:,0]*255, [], 'p1/'+str(pid)+'.png' )
        WriteImgKeys(bP[i,:,:,1]*255, [], 'p2/'+str(pid)+'.png' )
        GenAffineMapsImage(GTAivec, np.concatenate((vecE[8:16],vecE[0:8])), folder+'/map_'+str(pid)+'.png', GA, SP)        
        WriteImgKeys(DrawinP2(bP[i,:,:,0], bP[i,:,:,1], GTAi), [], folder+'/GT_'+str(pid)+'.png' )
        WriteImgKeys(DrawinP2(bP[i,:,:,0], bP[i,:,:,1], Ai), [], folder+'/GeoEsti_dropout_8pts_'+str(pid)+'.png' )
        
        pid += 1


def ComputeAffnetDataOnSimus(inputs, folder):
    bP, bGT = inputs
    h, w = 60, 60
    mask = np.zeros((h, w), np.uint8)
    mask[:] = 2.0
    SP = SquareOrderedPts(h,w,CV=False)
    
    emb_1, bP_Alist1 = AffNetHardNet_describe(np.expand_dims(np.array(bP[:,:,:,0]),axis=4))
    emb_2, bP_Alist2 = AffNetHardNet_describe(np.expand_dims(np.array(bP[:,:,:,1]),axis=4))

    pid = 0
    GA = GenAffine("", DryRun=True)
    for i in range(0,np.shape(bGT)[0]):
        GTA = GA.AffineFromNormalizedVector(bGT[i,:]) # transforms p2 into p1 coordinates
        GTAvec = affine_decomp(GTA)
        GTAi = cv2.invertAffineTransform( GTA )
        GTAivec = affine_decomp( GTAi )
        
        Ai = ComposeAffineMaps( bP_Alist2[i], cv2.invertAffineTransform(bP_Alist1[i]) )
        A = cv2.invertAffineTransform(Ai) # transforms p2 into p1 coordinates
        
        Avec = affine_decomp(A,doAssert=False)
        Aivec = affine_decomp(Ai,doAssert=False)

        
        WriteImgKeys(bP[i,:,:,0]*255, [], 'p1/'+str(pid)+'.png' )
        WriteImgKeys(bP[i,:,:,1]*255, [], 'p2/'+str(pid)+'.png' )
        WriteImgKeys(DrawinP2(bP[i,:,:,0], bP[i,:,:,1], GTAi), [], folder+'/GT_'+str(pid)+'.png' )
        WriteImgKeys(DrawinP2(bP[i,:,:,0], bP[i,:,:,1], Ai), [], folder+'/Affnet_'+str(pid)+'.png' )
        
        pid += 1



# set this flag to True in order to generate Figure 5
if True:
    inputs = []
    for file in glob.glob('./acc-test/*.txt'):
            pathway = './acc-test/' + os.path.basename(file)[:-4]
            inputs.append( load_acc_test_data(pathway, JitterAngle=10) )      

    ### Compute all necessary data on defined models
    info_all = []

    diffs_wrt_GT, score_TransitionTilt = ComputeModelData(geoesti_dropout, inputs, WasNetAffine = False)
    info_all.append([diffs_wrt_GT, score_TransitionTilt, 'LOCATE'])

    diffs_wrt_GT, score_TransitionTilt = ComputeAffNetData(inputs)
    info_all.append([diffs_wrt_GT, score_TransitionTilt, 'Affnet'])

    diffs_wrt_GT, score_TransitionTilt = ComputeIdentityData(inputs)
    info_all.append([diffs_wrt_GT, score_TransitionTilt, 'Identity'])

    ### Save to disks
    WriteHistos2Disk(info_all)


# set this flag to True in order to generate Figure 4
if False:
    GAval = GenAffine("./imgs-val/", save_path = "./db-gen-val-75/")
    for d in affine_generator(GAval, batch_num=32):
        ComputeModelDataOnSimus(geoesti_dropout, d[0], folder='LOCATE', WasNetAffine = False)
        ComputeAffnetDataOnSimus( d[0], folder='Affnet')
        break







