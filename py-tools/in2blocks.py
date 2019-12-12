import sys
sys.path.append(".")

from library import *
import time


while True:
   print("GAtrain")
   GAtrain = GenAffine("./imgs-train/", save_path = "./db-gen-train/")
   GAtrain.LastTimeDataChecked = 0
   GAtrain.ScatteredGenData_2_BlockData()

   print("GAval")
   GAval = GenAffine("./imgs-val/", save_path = "./db-gen-val/")
   GAval.LastTimeDataChecked = 0
   GAval.ScatteredGenData_2_BlockData()

   # print("GAtest")
   # GAval = GenAffine("./imgs-test/", save_path = "./db-gen-test/")
   # GAval.LastTimeDataChecked = 0
   # GAval.ScatteredGenData_2_BlockData()

   # Sleep 30 minutes
   time.sleep(60*30)

