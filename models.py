from tensorflow.compat.v1.keras import layers
from tensorflow.compat.v1.keras.models import Model
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras import backend as K


def create_model(input_shape, output_shape, model_name = 'DA_Pts_base', Norm='L2', resume = True, ResumeFile = ''):
    if model_name == 'DA_Pts_base':
        train_model = GeoEsti_CreateModel(input_shape, output_shape, Norm=Norm)
        path2weights = 'model-data/model.DA_Pts_base_L1_60.hdf5'
    elif model_name == 'DA_Pts_dropout':
        train_model = GeoEsti_CreateModel(input_shape, output_shape, Spatial_Dropout=True, Norm=Norm)
        path2weights = 'model-data/model.DA_Pts_dropout_L1_75.hdf5'
    elif model_name == 'DA_Pts_2xNeurons':
        train_model = GeoEsti_CreateModel(input_shape, output_shape, B5_FC1_neurons = 2048, Norm=Norm)
        path2weights = 'model-data/model.Pts_2xNeurons_L2.hdf5'
    elif model_name == 'DA_Aff_base':
        train_model = GeoEsti_CreateModel(input_shape, output_shape, Norm=Norm)
        path2weights = 'model-data/model.Aff_base_L1.hdf5'
    elif model_name == 'DA_Aff_dropout':
        train_model = GeoEsti_CreateModel(input_shape, output_shape, Spatial_Dropout=True, Norm=Norm)
        path2weights = 'model-data/model.Aff_dropout_L2.hdf5'
    elif model_name == 'DA_Aff_2xNeurons':
        train_model = GeoEsti_CreateModel(input_shape, output_shape, B5_FC1_neurons = 2048, Norm=Norm)
        path2weights = 'model-data/model.Aff_2xNeurons_L2.hdf5'

    elif model_name == 'GeoSimi_simCos':
        path2weights_GeoEsti = 'model-data/model.DA_Pts_base_L1.hdf5'
        train_model, sim_type = GeoSimi_CreateModel(input_shape, output_shape, similarity = 'simCos', path2weights_GeoEsti=path2weights_GeoEsti)        
        path2weights = 'model-data/model.GeoSimi_Pts_base_hinge.hdf5'
    elif model_name == 'GeoSimi_simCos_dropout':
        path2weights_GeoEsti = 'model-data/model.DA_Pts_dropout_L1.hdf5'
        train_model, sim_type = GeoSimi_CreateModel(input_shape, output_shape, similarity = 'simCos', path2weights_GeoEsti=path2weights_GeoEsti)        
        path2weights = 'model-data/model.GeoSimi_Pts_dropout_hinge.hdf5'        

    elif model_name == 'DAsimi_hinge':
        train_model, sim_type = DAsimi_CreateModel(input_shape, loss = 'hinge')
        path2weights = 'model-data/model.DAsimi_hinge.hdf5'
    elif model_name == 'DAsimi_hinge_dropout':
        train_model, sim_type = DAsimi_CreateModel(input_shape, loss = 'hinge', Spatial_Dropout=True)
        path2weights = 'model-data/model.DAsimi_hinge_dropout.hdf5'
    elif model_name == 'DAsimi_crossentropy':
        train_model, sim_type = DAsimi_CreateModel(input_shape, loss = 'cross-entropy')
        path2weights = 'model-data/model.DAsimi_crossentropy.hdf5'
    elif model_name == 'DAsimi_crossentropy_dropout':
        train_model, sim_type = DAsimi_CreateModel(input_shape, loss = 'cross-entropy', Spatial_Dropout=True)
        path2weights = 'model-data/model.DAsimi_crossentropy_dropout.hdf5'


    elif model_name == 'AID_simCos_base':
        train_model, sim_type = AID_CreateModel(input_shape, desc_dim = 128, desc_between_0_1 = False, similarity = 'simCos')
        path2weights = 'model-data/'
    elif model_name == 'AID_simCos_128Desc_1FC':
        train_model, sim_type = AID_CreateModel(input_shape, desc_dim = 128, B5_FC1_neurons=0, desc_between_0_1 = False, similarity = 'simCos')
        path2weights = 'model-data/'
    elif model_name == 'AID_simCos_128Desc_1FC_dropout':
        train_model, sim_type = AID_CreateModel(input_shape, desc_dim = 128, B5_FC1_neurons=0, Spatial_Dropout=True, desc_between_0_1 = False, similarity = 'simCos')
        path2weights = 'model-data/'
    elif model_name == 'AID_simCos_BigDesc':
        train_model, sim_type = AID_CreateModel(input_shape, BigDesc = True, similarity = 'simCos')
        path2weights = 'model-data/model.AID_simCos_BigDesc.hdf5'
    elif model_name == 'AID_simCos_BigDesc_dropout':
        train_model, sim_type = AID_CreateModel(input_shape, BigDesc = True, Spatial_Dropout=True, similarity = 'simCos')
        path2weights = 'model-data/model.AID_simCos_BigDesc_dropout.hdf5'
    elif model_name == 'AID_simCos_between01':
        train_model, sim_type = AID_CreateModel(input_shape, desc_dim = 128, desc_between_0_1 = True, similarity = 'simCos')
        path2weights = 'model-data/'
    elif model_name == 'AID_simCos_2xdescdim_between01':
        train_model, sim_type = AID_CreateModel(input_shape, desc_dim = 256, desc_between_0_1 = True, similarity = 'simCos')
        path2weights = 'model-data/'
    elif model_name == 'AID_simCos_2xdescdim': # this one was wrong all the time
        train_model, sim_type = AID_CreateModel(input_shape, desc_dim = 256, desc_between_0_1 = False, similarity = 'simCos')
        path2weights = 'model-data/'
    elif model_name == 'AID_simFC_diff': # became nan to soon
        train_model, sim_type = AID_CreateModel(input_shape, desc_dim = 128, desc_between_0_1 = False, similarity = 'simFC_diff')
        path2weights = 'model-data/'
    elif model_name == 'AID_simFC_concat':
        train_model, sim_type = AID_CreateModel(input_shape, desc_dim = 128, desc_between_0_1 = False, similarity = 'simFC_concat')
        path2weights = 'model-data/'
    elif model_name == 'AID_simFC_concat_BigDesc':
        train_model, sim_type = AID_CreateModel(input_shape, BigDesc = True, similarity = 'simFC_concat_BigDesc')
        path2weights = 'model-data/'
    else:
        train_model = None
        print('Error: '+model_name+" does not exist !")
        resume = False

    if ResumeFile!='':
        path2weights = ResumeFile
    if resume:
        train_model.load_weights(path2weights)
        print(path2weights)
    if model_name[0:3] == 'AID' or model_name[0:7] =='GeoSimi' or model_name[0:6] =='DAsimi':
        return train_model, sim_type
    else:
        return train_model


def CreateGeometricModel(input_shape,Spatial_Dropout,BN, trainit = True):
    trainitLayers = True
    trainitBN = True
    # Geometric Model
    in_net = layers.Input(shape=input_shape, name='input_patches')
    x = layers.Conv2D(64, (3, 3),
    padding='same',
    name='block1_conv1', trainable=trainitLayers)(in_net)
    if BN:
        x = layers.BatchNormalization(name='block1_BN1', trainable=trainitBN)(x)
    x = layers.Activation('relu', name='block1_relu1')(x)

    x = layers.Conv2D(64, (3, 3),
    padding='same',
    name='block1_conv2', trainable=trainitLayers)(x)
    if BN:
        x = layers.BatchNormalization(name='block1_BN2', trainable=trainitBN)(x)
    x = layers.Activation('relu', name='block1_relu2')(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)


    # Block 2
    x = layers.Conv2D(64, (3, 3),
    padding='same',
    name='block2_conv1', trainable=trainitLayers)(x)
    if BN:
        x = layers.BatchNormalization(name='block2_BN1', trainable=trainitBN)(x)
    x = layers.Activation('relu', name='block2_relu1')(x)

    x = layers.Conv2D(64, (3, 3),
    padding='same',
    name='block2_conv2', trainable=trainitLayers)(x)
    if BN:
        x = layers.BatchNormalization(name='block2_BN2', trainable=trainitBN)(x)
    x = layers.Activation('relu', name='block2_relu2')(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)


    # Block 3
    x = layers.Conv2D(128, (3, 3),
    padding='same',
    name='block3_conv1', trainable=trainitLayers)(x)
    if BN:
        x = layers.BatchNormalization(name='block3_BN1', trainable=trainitBN)(x)
    x = layers.Activation('relu', name='block3_relu1')(x)

    x = layers.Conv2D(128, (3, 3),
    padding='same',
    name='block3_conv2', trainable=trainitLayers)(x)
    if BN:
        x = layers.BatchNormalization(name='block3_BN2', trainable=trainitBN)(x)
    x = layers.Activation('relu', name='block3_relu2')(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)


    # Block 4
    x = layers.Conv2D(128, (3, 3),
    padding='same',
    name='block4_conv1', trainable=trainitLayers)(x)
    if BN:
        x = layers.BatchNormalization(name='block4_BN1', trainable=trainitBN)(x)
    x = layers.Activation('relu', name='block4_relu1')(x)

    x = layers.Conv2D(128, (3, 3),
    padding='same',
    name='block4_conv2', trainable=trainitLayers)(x)
    if BN:
        x = layers.BatchNormalization(name='block4_BN2', trainable=trainitBN)(x)
    if Spatial_Dropout:
        x = layers.SpatialDropout2D(rate=0.5,name='block4_Dropout1')(x)
    x = layers.Activation('relu', name='block4_relu2')(x)
    x = layers.Flatten(name='block5_flatten1')(x)

    geometric_model = Model(in_net, x, name='geometric_model')
    geometric_model.trainable = trainit
    return geometric_model


def GeoSimi_CreateModel(input_shape, output_shape, alpha_hinge = 0.2, Spatial_Dropout = False, BN = True, B5_FC1_neurons = 1024, similarity = 'simCos', verbose=True,  path2weights_GeoEsti=''):
    geometric_model_nontrainable = CreateGeometricModel(input_shape,Spatial_Dropout,BN, trainit=False)
    geo_dim = geometric_model_nontrainable.output_shape[1]
   
    # Similarity model    
    in_sim = layers.Input(shape=(geo_dim,), name='input_GeoInfo')
    x = layers.Dense(64,activation='relu',name='block1_FC1')(in_sim)
    x = layers.Dense(32,activation='relu',name='block1_FC2')(x)
    x = layers.Dense(1,activation='sigmoid',name='block1_FC3')(x)
    sim_model = Model(in_sim, x, name='similariy_model')
    in_net = layers.Input(shape=input_shape, name='input_patches')
    out_simi = sim_model(geometric_model_nontrainable(in_net))    
    GeoSimi_model = Model(in_net, out_simi, name='GeometricSimilarity')
    
    if similarity == 'simCos': # hinge loss
        # Similarity model
        in_p = layers.Input(shape=input_shape, name='input_patches_pos')
        in_n = layers.Input(shape=input_shape, name='input_patches_neg')        

        sim_type = 'inlist'
        out_net_positive = GeoSimi_model(in_p)
        out_net_negative = GeoSimi_model(in_n)

        class TopLossLayerClass(layers.Layer):
            def __init__(self, alpha = 0.2, **kwargs):
                self.alpha = alpha
                super(TopLossLayerClass, self).__init__(**kwargs)
            def call(self, inputs):
                out_net_positive, out_net_negative = inputs
                # Hinge loss computation
                loss = K.sum( K.maximum(out_net_negative - out_net_positive + self.alpha, 0) )#,axis=0)
                self.add_loss(loss)
                return loss
        TopLossLayer_obj = TopLossLayerClass(name='TopLossLayer', alpha = alpha_hinge)
        TopLossLayer = TopLossLayer_obj([out_net_positive, out_net_negative ])
        train_model = Model([in_p, in_n], TopLossLayer,name='TrainModel')

    # Mount the trained weights from the Geometric Estimator Network
    geoesti_model = GeoEsti_CreateModel(input_shape, output_shape, Spatial_Dropout, BN, B5_FC1_neurons, verbose=False)
    geoesti_model.load_weights(path2weights_GeoEsti) 
    train_model.get_layer("GeometricSimilarity").get_layer("geometric_model").set_weights(geoesti_model.get_layer("GeometricEstimator").get_layer("geometric_model").get_weights())

    if verbose:
        print('\n\n-------> The Geometric network architecture')
        GeoSimi_model.get_layer("geometric_model").summary()
        print('\n\n-------> The Similarity network architecture')
        sim_model.summary()
        print('\n\n-------> The GeometricSimilarity network architecture')
        GeoSimi_model.summary()
        print('\n\n-------> The Train architecture')
        train_model.summary()
    return train_model, sim_type


def GeoEsti_CreateModel(input_shape, output_shape, Spatial_Dropout = False, BN = True, B5_FC1_neurons = 1024, Norm = 'L2', verbose=True):
    ''' Geometric Estimator Model. 
    '''
    # Estimator Model
    geometric_model = CreateGeometricModel(input_shape,Spatial_Dropout,BN)
    geo_dim = geometric_model.output_shape[1]
    in_esti = layers.Input(shape=(geo_dim,), name='input_GeoInfo')
    x = in_esti 

    if B5_FC1_neurons>0:
        x = layers.Dense(B5_FC1_neurons,activation='relu',name='block5_FC1')(x)
    x = layers.Dense(output_shape[0],activation='sigmoid', name='block5_FC2')(x)
    estimator_model = Model(in_esti, x, name='estimator_model')
    in_net = layers.Input(shape=input_shape, name='input_patches')
    out_esti = estimator_model(geometric_model(in_net))
    GeoEsti_model = Model(in_net, out_esti, name='GeometricEstimator')            
    out_net = GeoEsti_model(in_net)

    # Groundtruth Model
    in_GT = layers.Input(shape=output_shape,name='input_GroundTruth')
    GT_model = Model(in_GT, in_GT, name='GroundTruth')
    out_GT = GT_model(in_GT)

    # TopLayer definition
    if Norm == 'L2':
        class TopLossLayerClass(layers.Layer):
            def __init__(self, **kwargs):
                super(TopLossLayerClass, self).__init__(**kwargs)
            def call(self, inputs):
                out_net,  out_GT = inputs
                loss =K.sum( K.sum( K.square( out_net - out_GT ), axis=-1 ) )
                self.add_loss(loss)
                return loss
    elif Norm == 'L1':
        class TopLossLayerClass(layers.Layer):
            def __init__(self, **kwargs):
                super(TopLossLayerClass, self).__init__(**kwargs)
            def call(self, inputs):
                out_net,  out_GT = inputs
                loss =K.sum( K.sum( K.abs( out_net - out_GT ), axis=-1 ) )
                self.add_loss(loss)
                return loss

    TopLossLayer_obj = TopLossLayerClass(name='TopLossLayer')

    # Train Model
    TopLossLayer = TopLossLayer_obj([out_net, out_GT ])
    train_model = Model([in_net, in_GT ], TopLossLayer,name='TrainModel')
    if verbose:
        print('\n\n-------> The GeometricEstimator network architecture')
        GeoEsti_model.summary()
        estimator_model.summary()
        print('\n\n-------> Train model connections')
        train_model.summary()
    return train_model



def CreateDescModel(input_shape, alpha_hinge, Spatial_Dropout, BN, B5_FC1_neurons, desc_dim, desc_between_0_1, BigDesc):
    # descriptor model
    in_desc = layers.Input(shape=input_shape, name='input_patches')

    x = layers.Conv2D(64, (3, 3),
    padding='same',
    name='block1_conv1')(in_desc)
    if BN:
        x = layers.BatchNormalization(name='block1_BN1')(x)
    x = layers.Activation('relu', name='block1_relu1')(x)

    x = layers.Conv2D(64, (3, 3),
    padding='same',
    name='block1_conv2')(x)
    if BN:
        x = layers.BatchNormalization(name='block1_BN2')(x)
    x = layers.Activation('relu', name='block1_relu2')(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)


    # Block 2
    x = layers.Conv2D(64, (3, 3),
    padding='same',
    name='block2_conv1')(x)
    if BN:
        x = layers.BatchNormalization(name='block2_BN1')(x)
    x = layers.Activation('relu', name='block2_relu1')(x)

    x = layers.Conv2D(64, (3, 3),
    padding='same',
    name='block2_conv2')(x)
    if BN:
        x = layers.BatchNormalization(name='block2_BN2')(x)
    x = layers.Activation('relu', name='block2_relu2')(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)


    # Block 3
    x = layers.Conv2D(128, (3, 3),
    padding='same',
    name='block3_conv1')(x)
    if BN:
        x = layers.BatchNormalization(name='block3_BN1')(x)
    x = layers.Activation('relu', name='block3_relu1')(x)

    x = layers.Conv2D(128, (3, 3),
    padding='same',
    name='block3_conv2')(x)
    if BN:
        x = layers.BatchNormalization(name='block3_BN2')(x)
    x = layers.Activation('relu', name='block3_relu2')(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)


    # Block 4
    x = layers.Conv2D(128, (3, 3),
    padding='same',
    name='block4_conv1')(x)
    if BN:
        x = layers.BatchNormalization(name='block4_BN1')(x)
    x = layers.Activation('relu', name='block4_relu1')(x)

    x = layers.Conv2D(128, (3, 3),
    padding='same',
    name='block4_conv2')(x)


    if BigDesc==False and BN:
        x = layers.BatchNormalization(name='block4_BN2')(x)

    if Spatial_Dropout:
        x = layers.SpatialDropout2D(rate=0.5,name='block4_Dropout1')(x)

    if BigDesc==False:
        x = layers.Activation('relu', name='block4_relu2')(x)


    # Block 5
    x = layers.Flatten(name='block5_flatten1')(x)

    if BigDesc==False:
        if B5_FC1_neurons>0:
            x = layers.Dense(B5_FC1_neurons,activation='relu',name='block5_FC1')(x)

        if desc_between_0_1:
            x = layers.Dense(desc_dim,activation='sigmoid',name='block5_FC2')(x)
        else:
            x = layers.Dense(desc_dim,name='block5_FC2')(x)

    desc_model = Model(in_desc, x, name='aff_desc')
    return desc_model


def AID_CreateModel(input_shape, alpha_hinge = 0.2, Spatial_Dropout = False, BN = True, B5_FC1_neurons = 1024, similarity = 'simCos', desc_dim = 128, desc_between_0_1 = False, BigDesc=False, verbose=True):

    desc_model = CreateDescModel(input_shape, alpha_hinge, Spatial_Dropout, BN, B5_FC1_neurons, desc_dim, desc_between_0_1, BigDesc)

    # similarity model
    if similarity[0:5] == 'simFC':
        if similarity[5:] == '_concat' or similarity[5:] == '_concat_BigDesc':
            sim_type = 'concat'
            desc_dim = 2*desc_model.output_shape[1]
        elif similarity[5:] == '_diff':
            sim_type = 'diff'
        # 2 siamese network
        in_desc1 = layers.Input(shape=input_shape, name='input_patches1')
        in_desc2 = layers.Input(shape=input_shape, name='input_patches2')
        emb_1 = desc_model(in_desc1)
        emb_2 = desc_model(in_desc2)

        # Similarity model
        in_sim = layers.Input(shape=(desc_dim,), name='input_diff_desc')
        x = layers.Dense(64,activation='relu',name='block1_FC1')(in_sim)
        x = layers.Dense(32,activation='relu',name='block1_FC2')(x)
        x = layers.Dense(1,activation='sigmoid',name='block1_FC3')(x)
        sim_model = Model(in_sim, x, name='sim')

        if sim_type == 'concat':
            x = layers.Concatenate(name='Concat')([emb_1, emb_2])
        else:
            x = layers.Subtract(name='Subtract')([emb_1, emb_2])

        out_net = sim_model(x)

        # Groundtruth Model
        in_GT = layers.Input(shape=(1,),name='input_GroundTruth')
        GT_model = Model(in_GT, in_GT, name='GroundTruth')
        out_GT = GT_model(in_GT)

        class TopLossLayerClass(layers.Layer):
            def __init__(self, **kwargs):
                super(TopLossLayerClass, self).__init__(**kwargs)
            def call(self, inputs):
                #out_net,  out_GT = inputs
                s,  t = inputs # t=1 -> Positive class, t=0 -> Negative class
                loss =K.sum(  t*K.log(s) + (1-t)*K.log(1-s) )
                self.add_loss(loss)
                return loss
        TopLossLayer_obj = TopLossLayerClass(name='TopLossLayer')

        TopLossLayer = TopLossLayer_obj([out_net, out_GT ])
        train_model = Model([in_desc1, in_desc2, in_GT], TopLossLayer,name='TrainModel')
    elif similarity == 'simCos': # hinge loss
        # Similarity model
        desc_dim = desc_model.output_shape[1]
        in_sim1 = layers.Input(shape=(desc_dim,), name='input_desc1')
        in_sim2 = layers.Input(shape=(desc_dim,), name='input_desc2')
        x = layers.Dot(axes=1, normalize=True, name='CosineProximity')([in_sim1,in_sim2]) # cosine proximity
        sim_model = Model([in_sim1,in_sim2], x, name='sim')

        # 3 siamese networks
        in_desc1 = layers.Input(shape=input_shape, name='input_patches_anchor')
        in_desc2 = layers.Input(shape=input_shape, name='input_patches_positive')
        in_desc3 = layers.Input(shape=input_shape, name='input_patches_negative')
        emb_1 = desc_model(in_desc1)
        emb_2 = desc_model(in_desc2)
        emb_3 = desc_model(in_desc3)
        sim_type = 'inlist'
        out_net_positive = sim_model([emb_1, emb_2])
        out_net_negative = sim_model([emb_1, emb_3])

        class TopLossLayerClass(layers.Layer):
            def __init__(self, alpha = 0.2, **kwargs):
                self.alpha = alpha
                super(TopLossLayerClass, self).__init__(**kwargs)
            def call(self, inputs):
                out_net_positive, out_net_negative = inputs
                # Hinge loss computation
                loss = K.sum( K.maximum(out_net_negative - out_net_positive + self.alpha, 0) )#,axis=0)
                self.add_loss(loss)
                return loss
        TopLossLayer_obj = TopLossLayerClass(name='TopLossLayer', alpha = alpha_hinge)
        TopLossLayer = TopLossLayer_obj([out_net_positive, out_net_negative ])
        train_model = Model([in_desc1, in_desc2, in_desc3], TopLossLayer,name='TrainModel')


    if verbose:
        print('\n\n-------> The network architecture for the affine descriptor computation !')
        desc_model.summary()
        print('\n\n-------> The network architecture for the similarity computation !')
        sim_model.summary()
        print('\n\n-------> Train model connections')
        train_model.summary()
    return train_model, sim_type



def DAsimi_CreateModel(input_shape, alpha_hinge = 0.1, Spatial_Dropout = False, BN = True, B5_FC1_neurons = 1024, loss = 'hinge', desc_dim = 0, desc_between_0_1 = False, verbose=True):

    desc_model = CreateDescModel(input_shape, alpha_hinge, Spatial_Dropout, BN, B5_FC1_neurons, desc_dim, desc_between_0_1, BigDesc=True)

    in_desc1 = layers.Input(shape=input_shape, name='input_patches')
    emb_1 = desc_model(in_desc1)        

    # Similarity model
    desc_dim = desc_model.output_shape[1]
    
    in_sim = layers.Input(shape=(desc_dim,), name='input_conv_desc')
    if B5_FC1_neurons>0:
        x = layers.Dense(B5_FC1_neurons,activation='relu',name='block1_FC1')(in_sim)
    # x = layers.Dense(32,activation='relu',name='block1_FC2')(x)
    x = layers.Dense(1,activation='sigmoid',name='block1_FC3')(x)
    sim_model = Model(in_sim, x, name='sim')

    out_sim = sim_model(emb_1)

    descsim_model = Model(in_desc1, out_sim, name='DescSimi')
    out_net = descsim_model(in_desc1)
    
    # similarity model
    if loss == 'cross-entropy':
        sim_type = 'diff'

        # Groundtruth Model
        in_GT = layers.Input(shape=(1,),name='input_GroundTruth')
        GT_model = Model(in_GT, in_GT, name='GroundTruth')
        out_GT = GT_model(in_GT)

        class TopLossLayerClass(layers.Layer):
            def __init__(self, **kwargs):
                super(TopLossLayerClass, self).__init__(**kwargs)
            def call(self, inputs):
                #out_net,  out_GT = inputs
                s,  t = inputs # t=1 -> Positive class, t=0 -> Negative class
                loss =K.sum(  t*K.log(s) + (1-t)*K.log(1-s) )
                self.add_loss(loss)
                return loss
        TopLossLayer_obj = TopLossLayerClass(name='TopLossLayer')

        TopLossLayer = TopLossLayer_obj([out_net, out_GT ])
        train_model = Model([in_desc1, in_GT], TopLossLayer,name='TrainModel')
    elif loss == 'hinge': # hinge loss
        sim_type = 'inlist'
        # siamese networks
        in_desc_p = layers.Input(shape=input_shape, name='input_patches_positive')
        out_net_positive = descsim_model(in_desc_p)

        in_desc_n = layers.Input(shape=input_shape, name='input_patches_negative')        
        out_net_negative = descsim_model(in_desc_n)

        class TopLossLayerClass(layers.Layer):
            def __init__(self, alpha = 0.2, **kwargs):
                self.alpha = alpha
                super(TopLossLayerClass, self).__init__(**kwargs)
            def call(self, inputs):
                out_net_positive, out_net_negative = inputs
                # Hinge loss computation
                loss = K.sum( K.maximum(out_net_negative - out_net_positive + self.alpha, 0) )#,axis=0)
                self.add_loss(loss)
                return loss
        TopLossLayer_obj = TopLossLayerClass(name='TopLossLayer', alpha = alpha_hinge)
        TopLossLayer = TopLossLayer_obj([out_net_positive, out_net_negative ])
        train_model = Model([in_desc_p, in_desc_n], TopLossLayer,name='TrainModel')


    if verbose:
        print('\n\n-------> The network architecture for the affine descriptor computation !')
        desc_model.summary()
        print('\n\n-------> The network architecture for the similarity computation !')
        sim_model.summary()
        print('\n\n-------> Train model connections')
        train_model.summary()
    return train_model, sim_type