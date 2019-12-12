MODEL_NAME = 'DA_Pts_dropout'
NORM = 'L1'
DegMax = 75
Debug = True
Parallel = False
ConstrastSimu = True # if True it randomly simulates contrast changes for each patch
DoBigEpochs = True


batch_number = 32
N_epochs = 5000
steps_epoch=100
NeededData = batch_number * N_epochs * steps_epoch + 1
SHOW_TB_weights = False # Show Net-weights info in TensorBoard


if MODEL_NAME[0:6]=="DA_Pts":
    NetAffine = False # if False the NeuralNet will estimate point coordinates
else:
    NetAffine = True # if True the NeuralNet will estimate the affine transformation itself

# When default GPU is being used... prepare to use a second one
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"


from library import *
from acc_test_library import *
import numpy as np
import time
import random
import cv2


def ProcessData(GA, stacked_patches, groundtruth_pts):
    if ConstrastSimu:
        channels = np.int32(np.shape(stacked_patches)[2]/2)
        val1 = random.uniform(1/3, 3)
        val2 = random.uniform(1/3, 3)
        # cv2.imwrite("/tmp/p1_before.png",stacked_patches[:,:,0]*255)
        # cv2.imwrite("/tmp/p2_before.png",stacked_patches[:,:,1]*255)
        for i in range(channels):
            stacked_patches[:,:,i] = np.power(stacked_patches[:,:,i],val1)
            stacked_patches[:,:,channels+i] = np.power(stacked_patches[:,:,channels+i],val2)
        # cv2.imwrite("/tmp/p1.png",stacked_patches[:,:,0]*255)
        # cv2.imwrite("/tmp/p2.png",stacked_patches[:,:,1]*255)
    if NetAffine:
        groundtruth_pts = GA.Nvec2Avec(groundtruth_pts)
        # groundtruth_pts2 = GA.Avec2Nvec(groundtruth_pts1)
        # print(GA.UnNormalizeVector(groundtruth_pts)-GA.UnNormalizeVector(groundtruth_pts2))
    return stacked_patches, groundtruth_pts #if ConstrastSimu==False -> Identity



GAval = GenAffine("./imgs-val/", save_path = "./db-gen-val-"+str(DegMax)+"/", DoBigEpochs = DoBigEpochs, tmax = DegMax)
GAtrain = GenAffine("./imgs-train/", save_path = "./db-gen-train-"+str(DegMax)+"/", DoBigEpochs = DoBigEpochs, tmax = DegMax)

Set_FirstThreadTouch(GAval,True)
Set_FirstThreadTouch(GAtrain,True)
stacked_patches, groundtruth_pts = GAtrain.gen_affine_patches()
stacked_patches, groundtruth_pts = ProcessData(GAtrain, stacked_patches, groundtruth_pts)



def affine_generator(GA, batch_num=32, Force2Gen=False, ForceFast=False):
    P_list = []
    GT_list = []
    FastThread = False
    t2sleep = 2*random.random()
    time.sleep(t2sleep)

    assert Force2Gen==False or ForceFast==False
    if ForceFast:
        FastThread = True

    if Force2Gen==False and Check_FirstThreadTouch(GA)==False:
        print("Fast Thread Created ! Needs "+str(NeededData)+" generated data")
        Set_FirstThreadTouch(GA,True)
        FastThread = True

    while True:
        if FastThread and ForceFast==False:
            GA.ScatteredGenData_2_BlockData() # it will be really done every 30 minutes

        stacked_patches, groundtruth_pts = [], []
        if FastThread and Force2Gen==False:
            # print(len(P_list), len(GT_list))
            stacked_patches, groundtruth_pts = GA.Fast_gen_affine_patches()
        else:
            stacked_patches, groundtruth_pts = GA.gen_affine_patches()

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
            if FastThread and Force2Gen==False:
                # print(len(P_list), len(GT_list))
                stacked_patches, groundtruth_pts = GA.Fast_gen_affine_patches()
            else:
                stacked_patches, groundtruth_pts = GA.gen_affine_patches()

            stacked_patches, groundtruth_pts = ProcessData(GA, stacked_patches, groundtruth_pts)
            bP[i,:,:,:] = stacked_patches
            bGT[i,:] = groundtruth_pts
        # print('These numbers should not repeat in other lines: '+ str(bP[0,0,0,0])+" "+str(bP[-1,0,0,0]))
        # print('Gen batch: '+str(np.shape(bP))+', '+str(np.shape(bGT)))
        yield [bP , bGT], None



#  VGG like network
from keras import layers
from keras.models import Model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(allow_soft_placement=True)
#, device_count = {'CPU' : 1, 'GPU' : 1})
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))


from models import *
vgg_input_shape = np.shape(stacked_patches)
vgg_output_shape = np.shape(groundtruth_pts)
train_model = create_model(vgg_input_shape, vgg_output_shape, model_name = MODEL_NAME, Norm=NORM, resume = False)




# ---> TRAIN NETWORK
import math
import scipy.special
import random
from sklearn.manifold import TSNE, MDS
from sklearn.metrics import f1_score, accuracy_score
from keras.callbacks import TerminateOnNaN, ModelCheckpoint, TensorBoard, LambdaCallback, ReduceLROnPlateau
import os
from shutil import copyfile
import matplotlib.pyplot as plt
plt.switch_backend('agg')
#modified from http://seoulai.com/2018/02/06/keras-and-tensorboard.html
class TensorboardKeras(object):
    def __init__(self, model, log_dir, GAval, GAtrain, static_val_num=1):
        self.model = model
        self.log_dir = log_dir
        self.session = K.get_session()
        self.lastloss = float('nan')
        self.lastvalloss = float('nan')
        self.GAval = GAval
        self.GAtrain = GAtrain
        self.static_Patches = []
        self.static_GTval = []
        self.static_val_num = static_val_num
        self.acc_data_inputs = []
        self.acc_data_names = []
        self.lastacc = 0
        self.TKid = random.randint(0,1000)

        for d in affine_generator(self.GAval, batch_num=self.static_val_num,ForceFast=True):
            self.static_Patches = d[0][0]
            self.static_GTval = d[0][1]
            break

        hs, ws = self.static_Patches.shape[1:3]
        self.SquarePatch = SquareOrderedPts(hs,ws,CV=False)

        self.static_val_repr = tf.placeholder(dtype=tf.float32)
        tf.summary.image("Repr/Static_validation", self.static_val_repr)

        self.dynamic_val_repr = tf.placeholder(dtype=tf.float32)
        tf.summary.image("Repr/Dynamic_validation", self.dynamic_val_repr)


        self.lr_ph = tf.placeholder(shape=(), dtype=tf.float32)
        tf.summary.scalar('Learning_rate', self.lr_ph)

        self.big_epoch = tf.placeholder(shape=(), dtype=tf.float32)
        tf.summary.scalar('Big_Epoch', self.big_epoch)

        self.val_loss_ph = tf.placeholder(shape=(), dtype=tf.float32)
        tf.summary.scalar('losses/validation', self.val_loss_ph)

        self.train_loss_ph = tf.placeholder(dtype=tf.float32)
        tf.summary.scalar('losses/training', self.train_loss_ph)

        # self.sift = cv2.xfeatures2d.SIFT_create( nfeatures = siftparams.nfeatures,
        # nOctaveLayers = siftparams.nOctaveLayers, contrastThreshold = siftparams.contrastThreshold,
        # edgeThreshold = siftparams.edgeThreshold, sigma = siftparams.sigma)
        self.global_acc_holder = tf.placeholder(dtype=tf.float32)
        tf.summary.scalar('accuracy/_GLOBAL_', self.global_acc_holder)

        self.acc_test_holder = []
        for file in glob.glob('./acc-test/*.txt'):
            self.acc_data_names.append( os.path.basename(file)[:-4] )
            i = len(self.acc_data_names) - 1
            pathway = './acc-test/' + self.acc_data_names[i]
            # asift_KPlist1, patches1, GT_Avec_list, asift_KPlist2, patches2 = load_acc_test_data(pathway)
            self.acc_data_inputs.append( load_acc_test_data(pathway) )

            self.acc_test_holder.append(tf.placeholder(dtype=tf.float32))
            self.acc_test_holder.append(tf.placeholder(dtype=tf.float32))
            self.acc_test_holder.append(tf.placeholder(dtype=tf.float32))
            self.acc_test_holder.append(tf.placeholder(dtype=tf.float32))
            self.acc_test_holder.append(tf.placeholder(dtype=tf.float32))
            self.acc_test_holder.append(tf.placeholder(dtype=tf.float32))
            self.acc_test_holder.append(tf.placeholder(dtype=tf.float32))
            self.variable_summaries(self.acc_test_holder[7*i  ], self.acc_data_names[i]+'-accuracy-info/zoom-diff')
            self.variable_summaries(self.acc_test_holder[7*i+1], self.acc_data_names[i]+'-accuracy-info/phi2-diff')
            self.variable_summaries(self.acc_test_holder[7*i+2], self.acc_data_names[i]+'-accuracy-info/tilt-diff')
            self.variable_summaries(self.acc_test_holder[7*i+3], self.acc_data_names[i]+'-accuracy-info/phi1-diff')
            tf.summary.scalar('accuracy/'+self.acc_data_names[i], self.acc_test_holder[7*i+4])
            self.variable_summaries(self.acc_test_holder[7*i+5], self.acc_data_names[i]+'-accuracy-info/tras-x_coor-diff')
            self.variable_summaries(self.acc_test_holder[7*i+6], self.acc_data_names[i]+'-accuracy-info/tras-y_coor-diff')

        if SHOW_TB_weights:
            l = np.shape(self.model.layers[2].get_weights())[0]
            self.weightsholder = []
            for i in range(0,l):
                self.weightsholder.append(tf.placeholder(dtype=tf.float32))
                self.variable_summaries(self.weightsholder[i], 'weights/'+repr(i).zfill(3)+'-layer')


        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_dir)

        copyfile(os.path.realpath(__file__), self.log_dir+"/"+os.path.basename(__file__))

    def _get_val_image_repr(self,batchGT,batchE, inSquare=True):
        fig = plt.figure()
        spn = self.GAval.NormalizeVector( Pts2Flatten(self.SquarePatch) )
        plt.plot(close_per(spn[0:8:2]),close_per(spn[1:8:2]),':k')
        for i in range(0,np.shape(batchGT)[0]):
            vec = batchGT[i,:]
            evec = batchE[i,:]
            if NetAffine:
                vec = self.GAval.Avec2Nvec(vec)
                evec = self.GAval.Avec2Nvec(evec)
            plt.plot(close_per(evec[0:8:2]),close_per(evec[1:8:2]),'-g')
            plt.plot(close_per(evec[8:16:2]),close_per(evec[9:16:2]),'--g')
            A = self.GAval.AffineFromNormalizedVector(evec)
            evec[0:8] = self.GAval.NormalizeVector( Pts2Flatten(AffineArrayCoor(self.SquarePatch,A)) )
            evec[8:16] = self.GAval.NormalizeVector( Pts2Flatten(AffineArrayCoor(self.SquarePatch,cv2.invertAffineTransform(A))) )
            plt.plot(close_per(vec[0:8:2]),close_per(vec[1:8:2]),'-b')
            plt.plot(close_per(vec[8:16:2]),close_per(vec[9:16:2]),'--b')
            plt.plot(close_per(evec[0:8:2]),close_per(evec[1:8:2]),'-r')
            plt.plot(close_per(evec[8:16:2]),close_per(evec[9:16:2]),'--r')

            # plt.plot(vec[::2],vec[1::2],'bx')
            # plt.plot(evec[::2],evec[1::2],'r+')
        if inSquare:
            plt.axis([0, 1, 0, 1])
        plt.title("Blue - GroundTruth / Red - Affine / Green - Homography")
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        plt.savefig('/tmp/val'+str(self.TKid)+'.png')
        plt.close(fig)
        img = load_image('/tmp/val'+str(self.TKid)+'.png')
        image = np.zeros(shape=(1,img.shape[0],img.shape[1],img.shape[2]))
        image[0,:,:,:] = (img).astype(np.float32)
        return image


    def variable_summaries(self,var,name):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope(name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)

    def _get_lr(self):
        return K.eval(self.model.optimizer.lr)

    def _get_weights(self,wpos):
        return self.model.layers[2].get_weights()[wpos]

    def on_epoch_end(self, epoch, logs):
        self.lastloss = np.ravel(logs['loss'])[0]
        self.lastvalloss = np.ravel(logs['val_loss'])[0]

    def on_epoch_begin(self, epoch, logs):
        for d in affine_generator(self.GAval, batch_num=self.static_val_num,ForceFast=True):
            dynamic_Eval = self.model.layers[2].predict(d[0][0])
            dynamic_GTval = d[0][1]
            break
        my_dict = {
                                                self.lr_ph: self._get_lr(),
                                                self.val_loss_ph: self.lastvalloss,
                                                self.big_epoch: get_big_epoch_number(self.GAtrain),
                                                self.train_loss_ph: self.lastloss,
                                                self.static_val_repr: self._get_val_image_repr(self.static_GTval, self.model.layers[2].predict(self.static_Patches), inSquare=True),
                                                self.dynamic_val_repr: self._get_val_image_repr(dynamic_GTval, dynamic_Eval, inSquare=False),                                              }
        if SHOW_TB_weights:
            l = np.shape(self.model.layers[2].get_weights())[0]
            for i in range(0,l):
                my_dict.update({self.weightsholder[i]: self._get_weights(i)})

        goodvec = []
        for i in range(0,len(self.acc_data_names)):
            diffs_GT, good = DA_ComputeAccuracy(self.GAval, self.model.layers[2], self.acc_data_inputs[i], WasNetAffine = NetAffine)
            diffs_GT = np.array(diffs_GT)
            my_dict.update({self.acc_test_holder[7*i  ]: diffs_GT[:,0]})
            my_dict.update({self.acc_test_holder[7*i+1]: diffs_GT[:,1]})
            my_dict.update({self.acc_test_holder[7*i+2]: diffs_GT[:,2]})
            my_dict.update({self.acc_test_holder[7*i+3]: diffs_GT[:,3]})
            my_dict.update({self.acc_test_holder[7*i+4]: good})
            my_dict.update({self.acc_test_holder[7*i+5]: diffs_GT[:,4]})
            my_dict.update({self.acc_test_holder[7*i+6]: diffs_GT[:,5]})
            goodvec.append(good)

        thisacc = np.mean(np.array(goodvec))
        if thisacc > self.lastacc:
            self.lastacc = thisacc
            self.model.save(self.log_dir+"/model.ckpt.max_acc.hdf5")
        my_dict.update({self.global_acc_holder: thisacc})
        summary = self.session.run(self.merged,
                                   feed_dict=my_dict)

        self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def on_epoch_end_cb(self):
        return LambdaCallback(on_epoch_end=lambda epoch, logs:
                                           self.on_epoch_end(epoch, logs))



from datetime import datetime

ts = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
log_path = "./summaries/" + MODEL_NAME + "_" + NORM + "_-_" + str(DegMax) + "deg_-_" + ts
tensorboard = TensorBoard(log_dir=log_path,
    write_graph=True, #This eats a lot of space. Enable with caution!
    #histogram_freq = 1,
    write_images=True,
    batch_size = 1,
    write_grads=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=25, verbose=1, mode='auto', cooldown=0, min_lr=0)

import keras
train_model.compile(loss=None, optimizer=keras.optimizers.Adam(lr=0.00001))
# loss_model_saver =  ModelCheckpoint(log_path + "/model.ckpt.min_loss.{epoch:04d}-{loss:.6f}.hdf5", monitor='loss', period=1, save_best_only=True)
loss_model_saver =  ModelCheckpoint(log_path + "/model.ckpt.min_loss.hdf5", monitor='loss', mode='min', period=1, save_best_only=True)
val_model_saver =  ModelCheckpoint(log_path + "/model.ckpt.min_val_loss.hdf5", monitor='val_loss', mode='min', period=1, save_best_only=True)
tboardkeras = TensorboardKeras(model=train_model, log_dir=log_path, GAval = GAval, GAtrain = GAtrain)
#on_epoch_begin or on_epoch_end
miscallbacks = [LambdaCallback(on_epoch_begin=lambda epoch, logs: tboardkeras.on_epoch_begin(epoch, logs),
                               on_epoch_end=lambda epoch, logs: tboardkeras.on_epoch_end(epoch, logs)),
                               tensorboard, TerminateOnNaN(), val_model_saver, loss_model_saver]#, reduce_lr]

Set_FirstThreadTouch(GAval,False)
Set_FirstThreadTouch(GAtrain,False)

if Debug:
    train_model.fit_generator(generator=affine_generator(GA=GAtrain,batch_num=2,ForceFast=True),
    validation_data=affine_generator(GA=GAval,batch_num=2,ForceFast=True), validation_steps=1,
    epochs=3, steps_per_epoch=2, callbacks = miscallbacks)
else:
    if Parallel:
        train_model.fit_generator(generator=affine_generator(GA=GAtrain,batch_num=batch_number,Force2Gen=True),
        validation_data=affine_generator(GA=GAval,batch_num=batch_number,Force2Gen=True), validation_steps=steps_epoch,
        epochs=N_epochs, steps_per_epoch=steps_epoch, callbacks = miscallbacks,
        max_queue_size=10,
        workers=6, use_multiprocessing=True)
    else:
        train_model.fit_generator(generator=affine_generator(GA=GAtrain,batch_num=batch_number,ForceFast=True),
        validation_data=affine_generator(GA=GAval,batch_num=batch_number,ForceFast=True), validation_steps=np.int32(steps_epoch/2),
        epochs=N_epochs, steps_per_epoch=steps_epoch, callbacks = miscallbacks)
