import numpy as np
import cv2
from matplotlib import pyplot as plt
from library import *
import time
import csv
import glob, os

class GTPairs(object):
    def __init__(self,name, query,target,T):
        self.pair_name = name
        self.query = query
        self.target = target
        self.Tmatrix = T 

class DatasetClass(object):
    def __init__(self,datasets_path, Ttype = 'Homography', name=''):
        self.name = name
        self.Ttype = Ttype
        self.path = datasets_path
        self.datapairs = []

def LoadDatasets():
    datasets = []

    # # EVD        
    # ds_path = 'acc-test/TestDatasets/EVD'
    # f = DatasetClass(ds_path, name = 'EVD')
    # for file in glob.glob(f.path+"/1/*"):
    #     f.datapairs.append( GTPairs(
    #         os.path.basename(file)[:-4],
    #         cv2.cvtColor( cv2.imread(os.path.join(ds_path,'1',os.path.basename(file)[:-4])+'.png') ,cv2.COLOR_BGR2GRAY).astype(np.uint8),
    #         cv2.cvtColor( cv2.imread(os.path.join(ds_path,'2',os.path.basename(file)[:-4])+'.png') ,cv2.COLOR_BGR2GRAY).astype(np.uint8),
    #         np.loadtxt(os.path.join(ds_path,'h',os.path.basename(file)[:-4]+'.txt'))
    #     ) )
    # datasets.append( f )

    # # OxAff        
    # ds_path = 'acc-test/TestDatasets/OxAff'
    # f = DatasetClass(ds_path, name = 'OxAff')
    # for tdir in glob.glob(f.path+"/*"):
    #     ext = glob.glob(tdir+"/img1.*")[0][-4:]
    #     for i in range(2,7):
    #         f.datapairs.append( GTPairs(
    #             os.path.basename(tdir)+'_1_to_'+str(i),
    #             cv2.cvtColor( cv2.imread(tdir+'/img1'+ext) ,cv2.COLOR_BGR2GRAY).astype(np.uint8),
    #             cv2.cvtColor( cv2.imread(tdir+'/img'+str(i)+ext) ,cv2.COLOR_BGR2GRAY).astype(np.uint8),
    #             np.loadtxt(tdir+'/H1to'+str(i)+'p')
    #         ) )
    # datasets.append( f )

    # # SymB        
    # ds_path = 'acc-test/TestDatasets/SymB'
    # f = DatasetClass(ds_path, name = 'SymB')
    # for tdir in glob.glob(f.path+"/*"):
    #     ext = glob.glob(tdir+"/01.*")[0][-4:]        
    #     f.datapairs.append( GTPairs(
    #         os.path.basename(tdir),
    #         cv2.cvtColor( cv2.imread(tdir+'/01'+ext) ,cv2.COLOR_BGR2GRAY).astype(np.uint8),
    #         cv2.cvtColor( cv2.imread(tdir+'/02'+ext) ,cv2.COLOR_BGR2GRAY).astype(np.uint8),
    #         np.loadtxt(tdir+'/H1to2')
    #     ) )
    # datasets.append( f )


    # # EF
    # ds_path = 'acc-test/TestDatasets/EF'
    # f = DatasetClass(ds_path, name = 'EF')
    # for tdir in glob.glob(f.path+"/*"):
    #     ext = glob.glob(tdir+"/img1.*")[0][-4:]
    #     for i in range(2,len(glob.glob(tdir+"/img*.png"))+1):
    #         f.datapairs.append( GTPairs(
    #             os.path.basename(tdir)+'_1_to_'+str(i),
    #             cv2.cvtColor( cv2.imread(tdir+'/img1'+ext) ,cv2.COLOR_BGR2GRAY).astype(np.uint8),
    #             cv2.cvtColor( cv2.imread(tdir+'/img'+str(i)+ext) ,cv2.COLOR_BGR2GRAY).astype(np.uint8),
    #             np.loadtxt(tdir+'/H1to'+str(i)+'p')
    #         ) )
    # datasets.append( f )

    # SIFT-AID        
    ds_path = 'acc-test/'
    f = DatasetClass(ds_path, name = 'SIFT-AID')
    f.datapairs.append( GTPairs('adam', cv2.cvtColor( cv2.imread(os.path.join(ds_path,'adam.1.png')) ,cv2.COLOR_BGR2GRAY).astype(np.uint8),
            cv2.cvtColor( cv2.imread(os.path.join(ds_path,'adam.2.png')) ,cv2.COLOR_BGR2GRAY).astype(np.uint8),
            np.loadtxt(os.path.join(ds_path,'adam.txt'))  ) )
    f.datapairs.append( GTPairs('arc', cv2.cvtColor( cv2.imread(os.path.join(ds_path,'arc.1.png')) ,cv2.COLOR_BGR2GRAY).astype(np.uint8),
            cv2.cvtColor( cv2.imread(os.path.join(ds_path,'arc.2.png')) ,cv2.COLOR_BGR2GRAY).astype(np.uint8),
            np.loadtxt(os.path.join(ds_path,'arc.txt'))  ) )
    f.datapairs.append( GTPairs('coca', cv2.cvtColor( cv2.imread(os.path.join(ds_path,'coca.1.png')) ,cv2.COLOR_BGR2GRAY).astype(np.uint8),
            cv2.cvtColor( cv2.imread(os.path.join(ds_path,'coca.2.png')) ,cv2.COLOR_BGR2GRAY).astype(np.uint8),
            np.loadtxt(os.path.join(ds_path,'coca.txt'))  ) )
    f.datapairs.append( GTPairs('graf', cv2.cvtColor( cv2.imread(os.path.join(ds_path,'graf.1.png')) ,cv2.COLOR_BGR2GRAY).astype(np.uint8),
            cv2.cvtColor( cv2.imread(os.path.join(ds_path,'graf.2.png')) ,cv2.COLOR_BGR2GRAY).astype(np.uint8),
            np.loadtxt(os.path.join(ds_path,'graf.txt'))  ) )
    f.datapairs.append( GTPairs('notredame', cv2.cvtColor( cv2.imread(os.path.join(ds_path,'notredame.1.png')) ,cv2.COLOR_BGR2GRAY).astype(np.uint8),
            cv2.cvtColor( cv2.imread(os.path.join(ds_path,'notredame.2.png')) ,cv2.COLOR_BGR2GRAY).astype(np.uint8),
            np.loadtxt(os.path.join(ds_path,'notredame.txt'))  ) )
    datasets.append( f )

    # # OxAff Viewpoint
    # ds_path = 'acc-test/TestDatasets/OxAff'
    # f = DatasetClass(ds_path, name = 'OxAff Viewpoint')
    # for tdir in [f.path+"/wall", f.path+"/graf"]:
    #     ext = glob.glob(tdir+"/img1.*")[0][-4:]
    #     for i in range(2,7):
    #         f.datapairs.append( GTPairs(
    #             os.path.basename(tdir)+'_1_to_'+str(i),
    #             cv2.cvtColor( cv2.imread(tdir+'/img1'+ext) ,cv2.COLOR_BGR2GRAY).astype(np.uint8),
    #             cv2.cvtColor( cv2.imread(tdir+'/img'+str(i)+ext) ,cv2.COLOR_BGR2GRAY).astype(np.uint8),
    #             np.loadtxt(tdir+'/H1to'+str(i)+'p')
    #         ) )
    # datasets.append( f )

    # # GDB        
    # ds_path = 'acc-test/TestDatasets/GDB'
    # f = DatasetClass(ds_path, name = 'GDB', Ttype = 'None')
    # for file in glob.glob(f.path+"/*"):
    #     f.datapairs.append( GTPairs(
    #         os.path.basename(file)[:-4],
    #         cv2.cvtColor( cv2.imread(os.path.join(ds_path+'/'+os.path.basename(file)[:-5]+'1'+os.path.basename(file)[-4:])) ,cv2.COLOR_BGR2GRAY).astype(np.uint8),
    #         cv2.cvtColor( cv2.imread(os.path.join(ds_path+'/'+os.path.basename(file)[:-5]+'2'+os.path.basename(file)[-4:])) ,cv2.COLOR_BGR2GRAY).astype(np.uint8),
    #         None
    #     ) )
    # datasets.append( f )

    return datasets



def DA_ComputeAccuracy(GA, model, inputs, WasNetAffine = True):
    asift_KPlist1, patches1, GT_Avec_list, asift_KPlist2, patches2 = inputs
    assert len(asift_KPlist1)==len(patches1)==len(GT_Avec_list)==len(asift_KPlist2)==len(patches2)

    patchshape = np.shape(patches1[0])
    bPshape = tuple([1]) + tuple( patchshape ) + tuple([2])
    bP = np.zeros(shape=bPshape, dtype = np.float32)
    Identity = np.float32([[1, 0, 0], [0, 1, 0]])
    Ivec = affine_decomp(Identity) # [1,0,1,0,0,0]
    good = 0
    diffs_GT = []
    for k in range(0,len(asift_KPlist1)):
        bP[0,:,:,:] = np.dstack((patches1[k]/GA.imgdivfactor, patches2[k]/GA.imgdivfactor))

        # Estimated
        Avec = model.predict(bP)
        if WasNetAffine:
            Avec = Avec[0]*GA.Avec_factor + GA.Avec_tras
            A = np.reshape( affine_decomp2affine( Avec[0:6] ), (2,3) ) # transforms p2 into p1 coordinates
        else:
            A = GA.AffineFromNormalizedVector( Avec[0] ) # transforms p2 into p1 coordinates
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

    return diffs_GT, np.float(good)/len(asift_KPlist1)

def load_acc_test_data(pathway, JitterAngle=0):
    img1 = cv2.cvtColor(cv2.imread(pathway+'.1.png'),cv2.COLOR_BGR2GRAY) # queryImage
    img2 = cv2.cvtColor(cv2.imread(pathway+'.2.png'),cv2.COLOR_BGR2GRAY) # trainImage

    H = np.loadtxt(pathway+'.txt')
    csvfile = open(pathway+'.csv', 'r')
    sr = csv.reader(csvfile, delimiter=',', quotechar='|')
    asift_KPlist1 = []
    asift_KPlist2 = []
    GT_Avec_list = []
    Identity = np.float32([[1, 0, 0], [0, 1, 0]])
    for row in sr:
        if len(row)==15 and row[0]!='x1':
            center = np.array([float(row[0]),float(row[1]),1]).reshape(3,1)
            Avec = affine_decomp(FirstOrderApprox_Homography(H,center))
            zoom_ratio = Avec[0]
            params = [[o1,l1,o2,l2] for o1 in range(0,3) for l1 in range(0, siftparams.nOctaveLayers+1) for o2 in range(0,3) for l2 in range(0, siftparams.nOctaveLayers+1)]
            mindiff = math.inf
            qmin = [0.0,0.0,0.0,0.0]
            for q in params:
                # we would like: zoom_ratio = zoom_1 / zoom_2
                temp = abs( zoom_ratio - (pow(2,-q[0])*pow(2.0, -q[1]/siftparams.nOctaveLayers))/(pow(2,-q[2])*pow(2.0, -q[3]/siftparams.nOctaveLayers)) )
                if mindiff>temp:
                    mindiff = temp
                    qmin = q

            q = qmin
            q_shift = 1
            temp = random.randint(q[1]-q_shift, q[1]+q_shift)
            while temp<0 or temp>siftparams.nOctaveLayers+1:
                temp = random.randint(q[1]-q_shift, q[1]+q_shift)
            q[1] = int(temp)
            temp = random.randint(q[3]-q_shift, q[3]+q_shift)
            while temp<0 or temp>siftparams.nOctaveLayers+1:
                temp = random.randint(q[3]-q_shift, q[3]+q_shift)
            q[3] = int(temp)

            lambda1 = pow(2,-q[0])*pow(2.0, -q[1]/siftparams.nOctaveLayers)
            angle1 = float(row[3])
            t1 = float(row[5])
            theta1 = np.deg2rad(float(row[6]))
            angle1 = angle1 if angle1>=0 else angle1+2*np.pi
            kp1 = cv2.KeyPoint(x = float(row[0]), y = float(row[1]),
                    _size = 2.0*siftparams.sigma*pow(2.0, q[1]/siftparams.nOctaveLayers)*pow(2.0,q[0]),
                    _angle = np.rad2deg(angle1),#+random.randint(0,10),
                    _response = 0.9, _octave = packSIFTOctave(q[0],q[1]),
                    _class_id = 0)
            A = cv2.invertAffineTransform( affine_decomp2affine( [1.0, 0.0, t1, theta1, 0.0, 0.0] ) )
            kp1 = AffineKPcoor([kp1],A, Pt_mod = False)[0] # it will only change the angle info
            h, w = img1.shape[:2]
            kp1, temp = Filter_Affine_In_Rect([kp1],Identity,[0,0],[w,h])

            lambda2 = pow(2,-q[2])*pow(2.0, -q[3]/siftparams.nOctaveLayers)
            angle2 = float(row[7+3])
            t2 = float(row[7+5])
            theta2 = np.deg2rad(float(row[7+6]))
            angle2 = angle2 if angle2>=0 else angle2+2*np.pi
            kp2 = cv2.KeyPoint(x = float(row[7+0]), y = float(row[7+1]),
                    _size = 2.0*siftparams.sigma*pow(2.0, q[3]/siftparams.nOctaveLayers)*pow(2.0,q[2]),
                    _angle = np.rad2deg(angle2),
                    _response = 0.9, _octave = packSIFTOctave(q[2],q[3]),
                    _class_id = 0)

            A = cv2.invertAffineTransform( affine_decomp2affine( [1.0, 0.0, t2, theta2, 0.0, 0.0] ) )
            kp2 = AffineKPcoor([kp2],A, Pt_mod = False)[0]
            h, w = img2.shape[:2]
            kp2, temp = Filter_Affine_In_Rect([kp2],Identity,[0,0],[w,h])

            if len(kp1)>0 and len(kp2)>0:
                # Uncomment this to discard angle info and set it randomly
                kp1[0].angle = random.randint(0,360)
                if JitterAngle>0:
                    kp2[0].angle = (kp1[0].angle + random.randint(-JitterAngle,JitterAngle) )%360
                else:
                    kp2[0].angle = (kp1[0].angle )%360
                kp2 = AffineKPcoor(kp2, affine_decomp2affine(Avec), Pt_mod = False)
                GT_Avec_list.append( Avec ) # Still needs to be modified as is from im1 to im2
                asift_KPlist1.append( kp1[0] )
                asift_KPlist2.append( kp2[0] )
    csvfile.close()

    pyr1 = buildGaussianPyramid( img1, siftparams.nOctaves + 2 )
    pyr2 = buildGaussianPyramid( img2, siftparams.nOctaves + 2 )

    patches1, A_list1, Ai_list1 = ComputePatches(asift_KPlist1,pyr1)
    patches2, A_list2, Ai_list2 = ComputePatches(asift_KPlist2,pyr2)
    assert len(patches1)==len(patches2)==len(asift_KPlist1)==len(asift_KPlist2)==len(GT_Avec_list)

    # Lets now make GT_Avec_list really go from patch1 to patch2
    for k in range(0,len(patches1)):
        Avec = GT_Avec_list[k]
        A = affine_decomp2affine(Avec)
        A = ComposeAffineMaps( A_list2[k], ComposeAffineMaps(A, Ai_list1[k]) )
        GT_Avec_list[k] = affine_decomp( A )
    return asift_KPlist1, patches1, GT_Avec_list, asift_KPlist2, patches2
