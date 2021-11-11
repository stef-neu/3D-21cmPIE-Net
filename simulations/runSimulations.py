import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import logging, sys, glob, random, optparse, os
import py21cmfast as p21c
from py21cmfast import cache_tools
import time

o = optparse.OptionParser()
o.set_usage('runSimulations.py [options] [N_lightcones]')

o.add_option('--astroOnly', dest='ao', default=False, action="store_true",
             help="Produces an astro-only dataset in a CDM universe by sampling LX,E0,Tvir and zeta (default: additional sampling of Omega_m and n_WDM).")

o.add_option('--filter', dest='filter', default=True, action="store_false",
             help="When set only lightcones that obey (observationally motivated) cuts in tau and global neutral fraction pass. For now, if a lightcone does not pass, the run will  be restarted with new random initial parameters.")

o.add_option('--threads', dest='threads', default=1, type=int,
             help="Number of threads used for simulations.")

o.add_option('--saliency', dest='sal', default=False, action="store_true",
             help="Use this flag to produce the light-cones required for paper_plots/saliency_maps.py. If this flag is set then N_lightcones specifies the number of lightcones with the SAME set of parameters")

opts, args = o.parse_args(sys.argv[1:])

logger = logging.getLogger('21cmFAST')
logger.setLevel(logging.INFO)
    
# Settings
p21c.inputs.global_params.P_CUTOFF = not opts.ao # CDM universe for astro-only dataset
height_dim = 140 # Number of pixels
box_len = 200 # Box size in Mpc
z_min = 5.0 # lowest redshift box
N_lc = int(args[0])
recalculate_redshifts = False # The redshifts associated with each box depend on the light-cone parameters, (e.g. box_len, height_dim, array of redshifts). For the given settings they are stored in redshifts5.npy
if isinstance(N_lc,int)==False:
   N_lc = 2
   print('N_lightcones was not given, setting to default 2')
   continue

if not recalculate_redshifts:
   with open("redshifts5.npy","rb") as data:
      redshifts=list(np.load(data,allow_pickle=True))
      redshifts.sort()

if opts.sal:
   # Produce light-cones for ../paper_plots/SaliencyMaps.py 
   dest="../paper_plots/input/"
   os.makedirs(dest, exist_ok=True)
   writer = tf.io.TFRecordWriter(dest+'bare_sim.tfrecord')
else:
   # Do not overwrite existing files 
   dest="output/"
   os.makedirs(dest, exist_ok=True)
   i=1
   while(dest+'run'+str(i)+'.tfrecord' in glob.glob(dest+"*")):
      i+=1
   writer = tf.io.TFRecordWriter(dest+'run'+str(i)+'.tfrecord')

j=0
os.makedirs("_cache", exist_ok=True)
while j<N_lc:
   # Cleanup
   cache_tools.clear_cache(direc="_cache")
   startTime = time.time()

   if opts.sal:
      # Produce light-cones for ../paper_plots/SaliencyMaps.py with the following parameters 
      WDM=2.
      OMm=0.31
      E0=500.
      LX=40.
      Tvir=4.7
      Zeta=30.
   else:
      # Random sampling over prior parameter ranges
      WDM=random.uniform(0.3,10.0)
      OMm=((0.02242 + 0.11933) / 0.6766 ** 2 if opts.ao else random.uniform(0.2,0.4))
      E0=random.uniform(100,1500)
      LX=random.uniform(38,42)
      Tvir=random.uniform(4,5.3)
      Zeta=random.uniform(10,250)
   
   print("lightcone No. = " + str(j))
   print("m_WDM = "+str(WDM))
   print("Omega_m = "+str(OMm))
   print("E_0 = "+str(E0))
   print("L_X = "+str(LX))
   print("T_vir = "+str(Tvir))
   print("zeta = "+str(Zeta))

   # Light-cone creation
   p21c.inputs.global_params.M_WDM = WDM
   lightcone = p21c.run_lightcone(
      redshift = z_min,
      cosmo_params = p21c.CosmoParams(OMm=OMm),
      astro_params = p21c.AstroParams(HII_EFF_FACTOR=Zeta,L_X=LX,NU_X_THRESH=E0,ION_Tvir_MIN=Tvir),
      user_params = {"HII_DIM": height_dim, "BOX_LEN": box_len,"PERTURB_ON_HIGH_RES":True,"N_THREADS":opts.threads,"USE_INTERPOLATION_TABLES": False},
      flag_options = {"USE_TS_FLUCT": True,"INHOMO_RECO":True},
      direc='_cache',
      write = recalculate_redshifts # Set to false to save disk space during runtime. Set to true if memory is not a concern. write=true is necessary when recalculate_redshifts=true
   )
   # Only required if light-cone specific parameters changed
   if recalculate_redshifts:
      redshifts = []
      for boxname in p21c.cache_tools.list_datasets(kind="BrightnessTemp",direc="_cache"):
         box=p21c.cache_tools.readbox(fname=boxname,direc="_cache",load_data=False)
         redshifts.append(box.redshift)
      redshifts.sort()
      recalculate_redshifts=False

   # Compute optical depth
   gxH=lightcone.global_xH
   gxH=gxH[::-1]
   tau=p21c.compute_tau(redshifts=redshifts,global_xHI=gxH)

   # Apply tau and global neutral fraction at z=5 (gxH[0]) filters
   if opts.filter and (tau>0.089 or gxH[0]>0.1):
      print("Filtered and restarted due to")
      print("tau="+str(tau))
      print("gxH(z=5)="+str(gxH[0]))
      continue

   # Save the light-cones to a tfrecords file
   attr=getattr(lightcone,"brightness_temp")
   lc_list = tf.train.BytesList(value=[attr[:,:,:2350].flatten().astype(np.float16).tobytes()])
   lb_list = tf.train.FloatList(value=[WDM,OMm,LX,E0,Tvir,Zeta])
   taulist = tf.train.FloatList(value=[tau])
   zlist = tf.train.FloatList(value=redshifts)
   gxHlist = tf.train.FloatList(value=gxH)
   image = tf.train.Feature(bytes_list=lc_list)
   label = tf.train.Feature(float_list=lb_list)
   tau = tf.train.Feature(float_list=taulist)
   redshift = tf.train.Feature(float_list=zlist)
   gxH = tf.train.Feature(float_list=gxHlist)
   lc_dict={
      'image': image,
      'label': label,
      'tau': tau,
      'z': redshift,
      'gxH': gxH,
   }
   features = tf.train.Features(feature=lc_dict)
   labeled_data = tf.train.Example(features=features)
   writer.write(labeled_data.SerializeToString())
   j+=1 
   executionTime = (time.time() - startTime)
   print('Execution time in seconds: ' + str(executionTime))

# Cleanup
writer.close()
cache_tools.clear_cache(direc="_cache")
