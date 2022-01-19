import numpy as np
import tensorflow as tf
import optparse, math, sys, glob, os
import py21cmfast as p21c

# The mock transformation logic of the light-cone boxes has been kindly provided by Caroline Heneka; it is based on an analogous transformation in C-code.

# Tool for efficient mock creation. Creates mocks for all light-cone files matching the pattern given by the data option and stores them in an output folder
o = optparse.OptionParser()
o.set_usage('create_mocks.py [options]')

o.add_option('--model', dest='model', default="opt",
             help="Foreground model to be used for mock creation. Options are opt and mod.")

o.add_option('--data', dest='data', default="../simulations/output/*.tfrecord",
             help="File pattern for the light-cone files")

o.add_option('--saliency', dest='sal', default=False, action="store_true",
             help="Use this flag to produce the light-cones required for paper_plots/SaliencyMaps.py. This changes both the input and output directories")

o.add_option('--astroOnly', dest='ao', default=False, action="store_true",
             help="Enables the creation of mocks for astro-only simulations")

opts, args = o.parse_args(sys.argv[1:])

# Read light-cones from tfrecords files
def parse_function(files):
    keys_to_features = {"label":tf.io.FixedLenFeature(((4 if opts.ao else 6)),tf.float32),
                        "image":tf.io.FixedLenFeature((),tf.string),
                        "tau":tf.io.FixedLenFeature((),tf.float32),
                        "gxH":tf.io.FixedLenFeature((92),tf.float32),
                        "z":tf.io.FixedLenFeature((92),tf.float32),}
    parsed_features = tf.io.parse_example(files, keys_to_features)
    image = tf.io.decode_raw(parsed_features["image"],tf.float16)
    image = tf.reshape(image,(140,140,2350))
    return image, tf.concat([parsed_features["label"],[parsed_features["tau"]],parsed_features["gxH"],parsed_features["z"]],axis=0) #m_WDM,Omega_m,L_X,E_0,T_vir,zeta,tau,gxH*92,z*92

# Save mock light-cones as a tfrecords file
def save(lightcone,labels,writer):
    lc_list = tf.train.BytesList(value=[lightcone.flatten().astype(np.float16).tobytes()])
    lb_list = tf.train.FloatList(value=labels[:(4 if opts.ao else 6)])
    taulist = tf.train.FloatList(value=[labels[(4 if opts.ao else 6)]])
    gxHlist = tf.train.FloatList(value=labels[(5 if opts.ao else 7):(97 if opts.ao else 99)])
    zlist = tf.train.FloatList(value=labels[(97 if opts.ao else 99):])
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

# Read output from 21cmSense and sort by frequency
if opts.model=="opt":
    files = glob.glob("calcfiles/opt_mocks/SKA1_Lowtrack_6.0hr_opt_0.*_LargeHII_Pk_Ts1_Tb9_nf0.52_v2.npz")
elif opts.model=="mod":
    files = glob.glob("calcfiles/mod_mocks/SKA1_Lowtrack_6.0hr_mod_0.*_LargeHII_Pk_Ts1_Tb9_nf0.52_v2.npz")
else:
    print("Please choose a valid foreground model")
    exit()

files.sort(reverse=True)

# List all simulated light-cone files
if opts.sal:
    bare_lc = glob.glob("../paper_plots/input/bare_sim.tfrecord")
else:
    bare_lc = glob.glob(opts.data)

os.makedirs("output",exist_ok=True)
 
# For each tfrecords file
for fl in bare_lc:
    if opts.sal:
        writer=tf.io.TFRecordWriter("../paper_plots/input/"+opts.model+"mocks.tfrecord")
    else:
        # Created mocks are found in output and have the same name as the file for the bare simulations
        name=fl.split("/")
        print("Creating mocks for"+str(fl))
        name="output/"+name[-1]
        writer = tf.io.TFRecordWriter(name)

    # Create mocks for the tfrecords file number N as specified by the user
    dataset = tf.data.TFRecordDataset(fl)
    dataset = dataset.map(parse_function,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # For each light-cone in the tfrecords file
    for ex in dataset.as_numpy_iterator():
        delta_T=ex[0]
        label=ex[1]
        box_redshifts=label[99:]
        
        # 21cmFAST functions are used to derive the redshift associated with each pixel
        cosmo_params = p21c.CosmoParams(OMm=label[1])
        astro_params = p21c.AstroParams(INHOMO_RECO=True)
        user_params = p21c.UserParams(HII_DIM=140, BOX_LEN=200)
        flag_options = p21c.FlagOptions()
        sim_lightcone=p21c.LightCone(5.,user_params,cosmo_params,astro_params,flag_options,0,{"brightness_temp":delta_T},35.05)
        redshifts=sim_lightcone.lightcone_redshifts
    
        # Calculate the length of each box. Each box is associated with a single redshift and uses the calc_sense output file for the respective frequency
        box_len=np.array([])
        y=0
        z=0
        for x in range(len(delta_T[0][0])):
            if redshifts[x]>(box_redshifts[y+1]+box_redshifts[y])/2:
                box_len=np.append(box_len,x-z)
                y+=1
                z=x
        box_len=np.append(box_len,x-z+1)
  
        # Split the light-cone into the respective boxes
        y=0
        delta_T_split=False
        for x in box_len:
            if delta_T_split is False:
                delta_T_split=[delta_T[:,:,int(y):int(x+y)]]
            else:
                delta_T_split.append(delta_T[:,:,int(y):int(x+y)])
            y+=x

        output=False
        cell_size=200/140
        hii_dim=140
        k140=np.fft.fftfreq(140,d=cell_size/2./np.pi)

        # For each box
        for x in range(len(box_len)):
            # Load 21cmSense output
            with np.load(files[x]) as data:
                ks = data["ks"]
                T_errs = data["T_errs"]

            # Calculate frequencies associated with the fourier transformation
            kbox=np.fft.rfftfreq(int(box_len[x]),d=cell_size/2./np.pi)
            volume=hii_dim*hii_dim*box_len[x]*cell_size**3
            err21a=np.zeros((140,140,int(box_len[x])))
            err21b=np.zeros((140,140,int(box_len[x])))

            # Create a fourier transformation of the simulation
            deldel_T = np.fft.rfftn(delta_T_split[x],s=(hii_dim,hii_dim,box_len[x]))

            # Draw random real and imaginary phases
            err21a = np.random.normal(loc=0.0,scale=1.0,size=(hii_dim,hii_dim,int(box_len[x])))
            err21b = np.random.normal(loc=0.0,scale=1.0,size=(hii_dim,hii_dim,int(box_len[x])))
            deldel_T_noise=np.zeros((hii_dim,hii_dim,int(box_len[x])),dtype=np.complex_)
            deldel_T_mock=np.zeros((hii_dim,hii_dim,int(box_len[x])),dtype=np.complex_)
      
            # Read in noise and interpolate it to the respective k-values
            for n_x in range(hii_dim):
                for n_y in range(hii_dim):
                    for n_z in range(int(box_len[x]/2+1)):
                        k_mag=math.sqrt(k140[n_x]**2+k140[n_y]**2+kbox[n_z]**2)
                        err21=np.interp(k_mag,ks,T_errs)

                        # Calculate and normalise the error in k space
                        if k_mag:
                            deldel_T_noise[n_x,n_y,n_z] = math.sqrt(math.pi*math.pi*volume/k_mag**3*err21)*(err21a[n_x,n_y,n_z]+err21b[n_x,n_y,n_z]*1j)
                        else:
                            deldel_T_noise[n_x,n_y,n_z]=0
        
                        # Add noise to signal in k space. uv_box is a mask of 0 or 1 to make it possible to potentially ignore diverging values or areas with very high errors
                        if(err21>=1000):
                            deldel_T_mock[n_x,n_y,n_z]=0
                        else:
                            deldel_T_mock[n_x,n_y,n_z]=deldel_T[n_x, n_y, n_z] + deldel_T_noise[n_x,n_y,n_z]/cell_size**3
            
            # Transform back into real space
            delta_T_mock=np.fft.irfftn(deldel_T_mock,s=(hii_dim,hii_dim,box_len[x]))

            # Rebuild the light-cone
            if output is False:
                output=delta_T_mock
            else:
                output=np.append(output,delta_T_mock,axis=2)
    
        save(output,label,writer)
    writer.close()
