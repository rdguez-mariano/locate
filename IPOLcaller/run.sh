#!/bin/bash
#/*
# * Copyright (c) 2020, Mariano Rodriguez <rdguez.mariano@gmail.com>
# * All rights reserved.
# *
# * This program is free software: you can use, modify and/or
# * redistribute it under the terms of the GNU General Public
# * License as published by the Free Software Foundation, either
# * version 3 of the License, or (at your option) any later
# * version. You should have received a copy of this license along
# * this program. If not, see <http://www.gnu.org/licenses/>.
# */

set -e

virtualenv=$1
demoextrasfolder=$2
binfolder=$3
input0="input_0.png"
input1="input_1.png"
gfilter=$4
aid_thres=$5
hardnet_thres=$6
detector=$7
descriptor=$8
ransac_iters=$9
precision=${10}


# Workaround for IPOL !
if [ -d $virtualenv ]; then
  source $virtualenv/bin/activate
  if python -c "import tensorflow" &> /dev/null; then
    TFv="`pip list | grep tensorflow | tail -n 1`"
    # echo "$TFv"
  else
    echo 'Installing tensorflow...'
    pip install --upgrade pip
    pip install --upgrade setuptools
    pip install --upgrade tensorflow==1.15.0 tensorflow-estimator==1.15.1
    pip install --upgrade h5py==2.7.1
  fi
fi

# echo "$virtualenv"
# echo "$binfolder"
# pwd

# ls -al $binfolder
# ls -al

locate_caller.py --im1 $input0 --im2 $input1 --gfilter $gfilter --aid_thres $aid_thres --hardnet_thres $hardnet_thres --detector $detector --descriptor $descriptor --workdir "./" --bindir $binfolder --visual --ransac_iters $ransac_iters --precision $precision

# ls -al


