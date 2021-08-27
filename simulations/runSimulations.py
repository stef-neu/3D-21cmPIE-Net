import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import logging, sys, glob, random, optparse, os
import py21cmfast as p21c
from py21cmfast import cache_tools

o = optparse.OptionParser()
o.set_usage('runSimulations.py [options] [N_runs]')

o.add_option('--astroOnly', dest='ao', default=False, action="store_true",
             help="Produces an astro-only dataset in a CDM universe by only sampling LX,E0,Tvir and zeta.")

o.add_option('--filter', dest='filter', default=True, action="store_false",
             help="If true, applies tau and global neutral fraction filters. If a run fails the filters it will be restarted with new initial parameters.")

o.add_option('--threads', dest='threads', default=1, type=int,
             help="Number of threads used for simulations.")

opts, args = o.parse_args(sys.argv[1:])

logger = logging.getLogger('21cmFAST')
logger.setLevel(logging.INFO)
    
# Settings
p21c.inputs.global_params.P_CUTOFF = not opts.ao
height_dim = 140 # Number of pixels in space dimensions
box_len = 200 # Box size in MPc
recalculate_redshifts = False # If you change any light-cone parameters (e.g. box_len, height_dim, redshift, max_redshift) recalculate_redshifts should be set to true

if not recalculate_redshifts:
   with open("redshifts5.npy","rb") as data:
      redshifts=list(np.load(data,allow_pickle=True))
      redshifts.sort()
  
# Do not overwrite existing files
i=1
if opts.ao:
   dest="output/AstroOnly/"
else:
   dest="output/FullPara/"
os.makedirs(dest, exist_ok=True)
while(dest+'Run'+str(i)+'.tfrecord' in glob.glob(dest+"*")):
   i+=1
writer = tf.io.TFRecordWriter(dest+'Run'+str(i)+'.tfrecord')

j=0
os.makedirs("_cache", exist_ok=True)
while j<int(args[0]):
   # Cleanup
   cache_tools.clear_cache(direc="_cache")

   # Random sampling over parameter ranges
   WDM=random.uniform(0.3,10.0)
   OMm=((0.02242 + 0.11933) / 0.6766 ** 2 if opts.ao else random.uniform(0.2,0.4))
   E0=random.uniform(100,1500)
   LX=random.uniform(38,42)
   Tvir=random.uniform(4,5.3)
   Zeta=random.uniform(10,250)
   p21c.inputs.global_params.M_WDM = WDM
   print("run number = " + str(j))
   print("m_WDM = "+str(WDM))
   print("OMm = "+str(OMm))
   print("E0 = "+str(E0))
   print("LX = "+str(LX))
   print("Tvir = "+str(Tvir))
   print("Zeta = "+str(Zeta))
   # Light-cone creation
   lightcone = p21c.run_lightcone(
      redshift = 5.0,
      cosmo_params = p21c.CosmoParams(OMm=OMm),
      astro_params = p21c.AstroParams(HII_EFF_FACTOR=Zeta,L_X=LX,NU_X_THRESH=E0,ION_Tvir_MIN=Tvir),
      user_params = {"HII_DIM": height_dim, "BOX_LEN": box_len,"PERTURB_ON_HIGH_RES":True,"N_THREADS":opts.threads,"USE_INTERPOLATION_TABLES": False},
      flag_options = {"USE_TS_FLUCT": True,"INHOMO_RECO":True},
      direc='_cache',
      write = recalculate_redshifts #Preventing the program from saving huge amounts of data to the disk during runtime. Set to true if memory is a concern.
   )
   
   # Only required if light-cone specific parameters were changed
   if recalculate_redshifts:
      redshifts = []
      for boxname in p21c.cache_tools.list_datasets(kind="BrightnessTemp",direc="_cache"):
         box=p21c.cache_tools.readbox(fname=boxname,direc="_cache",load_data=False)
         redshifts.append(box.redshift)
      redshifts.sort()
      recalculate_redshifts=False

   # computing tau=optical debth to reionization
   gxH=lightcone.global_xH
   gxH=gxH[::-1]
   tau=p21c.compute_tau(redshifts=redshifts,global_xHI=gxH)

   # Applying tau and gxH filters
   if opts.filter and (tau>0.089 or gxH[0]>0.1):
      print("Filtered and restarted due to")
      print("tau="+str(tau))
      print("gxH(z=5)="+str(gxH[0]))
      continue

   # Saving the light-cones to a tfrecords file
   attr=getattr(lightcone,"brightness_temp")
   LClist = tf.train.BytesList(value=[attr[:,:,:2350].flatten().astype(np.float16).tobytes()])
   LBlist = tf.train.FloatList(value=[WDM,OMm,LX,E0,Tvir,Zeta])
   Taulist = tf.train.FloatList(value=[tau])
   Zlist = tf.train.FloatList(value=redshifts)
   gxHlist = tf.train.FloatList(value=gxH)
   image = tf.train.Feature(bytes_list=LClist)
   label = tf.train.Feature(float_list=LBlist)
   Tau = tf.train.Feature(float_list=Taulist)
   Redshift = tf.train.Feature(float_list=Zlist)
   gxH = tf.train.Feature(float_list=gxHlist)
   LCdict={
      'image': image,
      'label': label,
      'tau': Tau,
      'z': Redshift,
      'gxH': gxH,
   }
   features = tf.train.Features(feature=LCdict)
   labeled_data = tf.train.Example(features=features)
   writer.write(labeled_data.SerializeToString())
   j+=1 

# Cleanup
writer.close()
cache_tools.clear_cache(direc="_cache")
