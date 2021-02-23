# LOCATE : a LOCal Affine Transform Estimator

This repository implements LOCATE, a method for retrieving local affine approximations between two images. Two main applications are aimed: Guided Matching and Homography estimation. The companion paper can be found [here](https://rdguez-mariano.github.io/pages/locate).

## Prerequisites

In order to quickly start using LOCATE we advise you to install [anaconda](https://www.anaconda.com/distribution/) and follow this guide.

##### Creating a conda environment for LOCATE

```bash
conda create --name locate python=3.5.4

source activate locate

pip install --upgrade pip
pip install -r requirements.txt
```

##### Compiling the C++ library

```bash
mkdir -p build && cd build && cmake .. && make && mv libDA.so ..
```

##### Possible install errors

If AttributeError: module 'cv2.cv2' has no attribute 'xfeatures2d' reinstall opencv-contrib

```bash
pip uninstall opencv-contrib-python
pip install opencv-python==3.4.2.16
```

#### Uninstall the LOCATE environment

If you want to remove this project from your computer just do:

```bash
conda deactivate
conda-env remove -n locate
rm -R /path/to/locate
```

## Reproducing results from the [companion paper](https://rdguez-mariano.github.io/pages/locate)

Recreate images from Figure 1, 4, 5 and 6:

```bash
mkdir -p temp
python py-tools/gen-Fig.1-WACV20.py
python py-tools/gen-Fig.4-5-WACV20.py
python py-tools/gen-GuidedMatching-Teaserimages-WACV20.py
```

Generate tables 1 and 2:

```bash
python py-tools/gen-tables-WACV20.py
```

## Features in this source code

Available matching methods (detectors and descriptors):
- SIFT + AID
- HessAff + AID
- HessAff + Affnet + HardNet
- SIFT + Affnet + HardNet
- HessianLaplace + RootSIFT
- SIFT + RootSIFT

Available geometric filters:
- USAC Homography
- USAC Fundamental
- ORSA Homography
- ORSA Fundamental
- RANSAC Baseline (for Homography)
- RANSAC_2pts (for Homography)
- RANSAC_affine (for Homography)

Let's take a look at some possible configurations for different methods.

```python
import cv2
from libLocalDesc import *
img1 = cv2.cvtColor(cv2.imread('./acc-test/adam.1.png'),cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(cv2.imread('./acc-test/adam.2.png'),cv2.COLOR_BGR2GRAY)

# HessAff + AID + USAC Homography
_, good_HC, _,_,_, ET_KP, ET_M = HessAffAID(img1,img2, MatchingThres = 4000, GFilter='USAC_H')
print("Hessaff-AID + RANSAC USAC --> FilteredMatches = %d, KeypointsTime = %3.3f, MatchingTime = %3.3f" %(len(good_HC),ET_KP,ET_M))

# HessAff + Affnet + HardNet + RANSAC_affine with affine maps provided by Affnet
_, good_HC, _,_,_, ET_KP, ET_M = HessAff_HardNet(img1,img2,GFilter='Aff_H-2',HessAffNet=True)
print("HessAffnet + RANSAC_affine with Affnet --> FilteredMatches = %d, KeypointsTime = %3.3f, MatchingTime = %3.3f" %(len(good_HC),ET_KP,ET_M))

# SIFT + RootSIFT + ORSA Homography
_, good_HC, KPlist1, KPlist2, H_sift, ET_KP, ET_M = RootSIFT(img1,img2, MatchingThres = 0.8, GFilter='ORSA_H')
print("RootSIFT + RANSAC_affine with Affnet --> FilteredMatches = %d, KeypointsTime = %3.3f, MatchingTime = %3.3f" %(len(good_HC),ET_KP,ET_M))

# HessianLaplace + RootSIFT + RANSAC_2pts with affine maps provided by HessianLaplace
_, good_HC, KPlist1, KPlist2, H_sift, ET_KP, ET_M = HessianLaplaceRootSIFT(img1,img2, MatchingThres = 0.8, GFilter='Aff_H-1')
print("HessianLaplace-RootSIFT + RANSAC_affine with HessianLaplace --> FilteredMatches = %d, KeypointsTime = %3.3f, MatchingTime = %3.3f" %(len(good_HC),ET_KP,ET_M))
```

## Using LOCATE for homography estimation

LOCATE can be used to retrieve tangent planes at verified matching points of the global mapping. This, as explained in the [companion paper](https://rdguez-mariano.github.io/pages/locate), can be used to build affine-aware RANSACs such as: RANSAC_2pts and RANSAC_affine.

Let us now execute the three equally structured RANSACs appearing in the companion paper: RANSAC baseline, RANSAC_2pts and RANSAC_affine.

```python
import cv2
from libLocalDesc import *
img1 = cv2.cvtColor(cv2.imread('./acc-test/adam.1.png'),cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(cv2.imread('./acc-test/adam.2.png'),cv2.COLOR_BGR2GRAY)

# SIFT-AID + RANSAC baseline
_, good_HC, KPlist1, KPlist2, H_AID, ET_KP, ET_M = siftAID(img1,img2, MatchingThres = 4000, Simi = 'SignProx', GFilter='Aff_H-0')
print("SIFT-AID + RANSAC baseline --> FilteredMatches = %d, KeypointsTime = %3.3f, MatchingTime = %3.3f" %(len(good_HC),ET_KP,ET_M))

# SIFT-AID + RANSAC_2pts with LOCATE
_, good_HC, KPlist1, KPlist2, H_AID, ET_KP, ET_M = siftAID(img1,img2, MatchingThres = 4000, Simi = 'SignProx', GFilter='Aff_H-1')
print("SIFT-AID + RANSAC_2pts with LOCATE --> FilteredMatches = %d, KeypointsTime = %3.3f, MatchingTime = %3.3f" %(len(good_HC),ET_KP,ET_M))

# SIFT-AID + RANSAC_affine with LOCATE
_, good_HC, KPlist1, KPlist2, H_AID, ET_KP, ET_M = siftAID(img1,img2, MatchingThres = 4000, Simi = 'SignProx', GFilter='Aff_H-2')
print("SIFT-AID + RANSAC_affine with LOCATE --> FilteredMatches = %d, KeypointsTime = %3.3f, MatchingTime = %3.3f" %(len(good_HC),ET_KP,ET_M))
```

LOCATE can also be used with other descriptors, but they should be based on the SIFT detector. For example, SIFT+Affnet+Hardnet could use approximating affine maps from either Affnet or LOCATE.

```python
# SIFT + Affnet + HardNet + RANSAC_affine with affine maps provided by Affnet
_, good_HC, KPlist1, KPlist2, H_sift, ET_KP, ET_M = SIFT_AffNet_HardNet(img1,img2, MatchingThres = 0.8, GFilter='Aff_H-N-2')
print("SIFT-Affnet + RANSAC_affine with Affnet --> FilteredMatches = %d, KeypointsTime = %3.3f, MatchingTime = %3.3f" %(len(good_HC),ET_KP,ET_M))

# SIFT + Affnet + HardNet + RANSAC_affine with affine maps provided by LOCATE
_, good_HC, KPlist1, KPlist2, H_sift, ET_KP, ET_M = SIFT_AffNet_HardNet(img1,img2, MatchingThres = 0.8, GFilter='Aff_H-2')
print("SIFT-Affnet + RANSAC_affine with LOCATE --> FilteredMatches = %d, KeypointsTime = %3.3f, MatchingTime = %3.3f" %(len(good_HC),ET_KP,ET_M))
```

More available configurations can be found in [gen-tables-WACV20.py](py-tools/gen-tables-WACV20.py).

## Using LOCATE for guided matching

This source code for guided matching is far from optimised. It was rather used for academic purposes. Our lazy implementation can be drastically improved to obtain better time performances.

Let's launch SIFT-AID with LOCATE guided matching (4 iterations).

```python
import cv2
from libLocalDesc import *
img1 = cv2.cvtColor(cv2.imread('./acc-test/adam.1.png'),cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(cv2.imread('./acc-test/adam.2.png'),cv2.COLOR_BGR2GRAY)

_, good_HC, KPlist1, KPlist2, H_AID, ET_KP, ET_M = siftAID(img1,img2, MatchingThres = 4000, Simi = 'SignProx', RegionGrowingIters=4)
print("Guided SIFT-AID with LOCATE --> FilteredMatches = %d, KeypointsTime = %3.3f, MatchingTime = %3.3f" %(len(good_HC),ET_KP,ET_M))
```

More available configurations can be found in [gen-tables-WACV20.py](py-tools/gen-tables-WACV20.py).

## Training the LOCATE network

##### Creating the needed directory structure

For training you need to specify three image datasets. These images will be used to generate optical affine views. Create default folders by typing:

```bash
mkdir -p imgs-train && mkdir -p imgs-val && mkdir -p imgs-test
```

Now, for example, you can download correspondent datasets from [MS-COCO](http://cocodataset.org) into the default folders.

Also, we need to create the following folders in order to storage tensorboard summaries and the resulting output images and data:

```bash
mkdir -p summaries && mkdir -p temp
```

##### Training

Once image datasets are available in *imgs-train* and *imgs-val* you can train the network.

```bash
python GeoEsti-train-model.py
```

Generated pairs of patches will be saved into their respective folders, e.g. (*db-gen-train-60*, *db-gen-val-60*). Those folders will have scattered files corresponding to patchs pairs. To create blocks of data that can be quickly (and automatically) reused by the trainer please launch also:

```bash
python py-tools/in2blocks.py
```

##### Some key variables in GeoEsti-train-model.py

Maximal viewpoint angle of the affine maps to be optically simulated.

```python
DegMax = 75
```

Set this variable to `False` if you want to really train. Use for test when modifying the code.

```python
Debug = True
```

Set this to `True` if no blocks of patch-data have been created yet. It will use all your CPU power for affine simulations. Deactivate back when you have created a sufficient amount of data blocks with [in2blocks.py](py-tools/in2blocks.py).

```python
Parallel = False
```

Set this variable to `True` if you want to cycle over blocks of patch-data.

```python
DoBigEpochs = True
```

Depending on you GPU, select the percentage of GPU memory to be used when training.

```python
config.gpu_options.per_process_gpu_memory_fraction = 0.3
```

## Summaries with tensorboard

Show all images (slider step = 1)

```bash
tensorboard --logdir="./" --samples_per_plugin="images=0"
```

If tensorboard crashes, try reinstalling it first !!! If locale error (unsupported locale) setting, do:

```bash
export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"
sudo dpkg-reconfigure locales
```

## Authors

* **Mariano Rodríguez** - [web page](https://rdguez-mariano.github.io/)
* **Gabriele Facciolo**
* **Rafael Grompone Von Gioi**
* **Pablo Musé**
* **Julie Delon** - [web page](https://delon.wp.imt.fr/)


## License

The code is distributed under the permissive MIT License - see the [LICENSE](LICENSE) file for more details.

## Acknowledgements

##### This project can optionally

* call libOrsa, libMatch and libNumerics. Copyright (C) 2007-2010, Lionel Moisan, distributed under the BSD license.
* call libUSAC. Copyright (c) 2012 University of North Carolina at Chapel Hill / See its [web page](http://www.cs.unc.edu/~rraguram/usac/) to see their specific associated licence.
* call Affnet. Copyright (c) 2018 Dmytro Mishkin / See its [source code](https://github.com/ducha-aiki/affnet) distributed under the permissive MIT License.
* call VLfeat. Copyright (C) 2007-11, Andrea Vedaldi and Brian Fulkerson, distributed under the BSD 2-Clause "Simplified" License. See its [source code](https://github.com/vlfeat/vlfeat) for more information.
* call SIFT-AID. Copyright (c) 2019 Mariano Rodriguez / See its [source code](https://github.com/rdguez-mariano/sift-aid) distributed under the permissive MIT License.

## Github repository

<https://github.com/rdguez-mariano/locate>
