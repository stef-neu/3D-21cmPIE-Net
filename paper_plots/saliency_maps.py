import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from matplotlib.ticker import AutoLocator
from matplotlib.transforms import Bbox
import py21cmfast as p21c
from py21cmfast import plotting
from tf_keras_vis.saliency import Saliency
import os

# Change the sigmoid activation of the last layer to a linear one. This is required for creating saliency maps
def model_modifier(m):
    m.layers[-1].activation = tf.keras.activations.linear
    return m

# Read light-cones from tfrecords files
def parse_function(files):
    keys_to_features = {"label":tf.io.FixedLenFeature((6),tf.float32),
                        "image":tf.io.FixedLenFeature((),tf.string),
                        "tau":tf.io.FixedLenFeature((),tf.float32),
                        "gxH":tf.io.FixedLenFeature((92),tf.float32),
                        "z":tf.io.FixedLenFeature((92),tf.float32),}
    parsed_features = tf.io.parse_example(files, keys_to_features)
    image = tf.io.decode_raw(parsed_features["image"],tf.float16)
    image = tf.reshape(image,(140,140,2350))
    return image, parsed_features["label"] # Image, m_WDM,Omega_m,L_X,E_0,T_vir,zeta

# Plot the saliency maps over the light-cones. Axis scales depend on the value of Omega_m. This function expects all light-cones to have the same Omega_m
def plot(filename,sim_lightcone,mock_lightcone,parameters,saliency_maps=False):
    # Define parameter names, ranges and latex code
    parameter_list=[["WDM",0.3,10,"$m_{WDM}$"],["OMm",0.2,0.4,"$\Omega_m$"],["LX",38,42,"$L_X$"],["E0",100,1500,"$E_0$"],["Tvir",4,5.3,"$T_{vir}$"],["Zeta",10,250,"$\zeta$"]]
    fig, ax = plt.subplots(
        2*len(parameters),
        1,
        sharex=False,
        sharey=False,
        figsize=(
            2350 * (0.007) + 0.5 ,
            140*0.02*len(parameters)
        ),
    )

    # Create opt mock and bare simulation saliency maps for each of the requested parameters
    for x,para in enumerate(parameters+parameters):
        # Plot bare simulations in the first half
        if x<len(parameters):
            fig, ax[x]=plotting.lightcone_sliceplot(sim_lightcone,slice_axis=0,slice_index=70,fig=fig,ax=ax[x],zticks="frequency")
            ax[x].images[-1].colorbar.remove()

        # Plot opt mocks in the second half
        else:
            fig, ax[x]=plotting.lightcone_sliceplot(mock_lightcone,slice_axis=0,slice_index=70,fig=fig,ax=ax[x])
            ax[x].images[-1].colorbar.remove()

        # Plot saliency maps
        if saliency_maps is not False:
            extent = (
                0,
                2350*200/140,                                                                            
                0,
                200,
            )
            ax[x].imshow(saliency_maps[x],origin="lower",cmap=cm.hot,alpha=0.7,extent=extent)

        # Adjust the design
        if x>0 and x<2*len(parameters)-1:
            ax[x].set_xticks([])
            ax[x].set_xlabel("")
            ax[x].tick_params(labelsize=12)
        ax[x].text(10,10,"$\delta "+parameter_list[para][3][1:],color="w",fontsize=14)
        ax[x].set_ylabel("")
    fig.text(0.01,0.62+0.02*len(parameters),"y-axis [Mpc]",rotation="vertical",fontsize=12)
    fig.text(0.01,0.23-0.005*len(parameters),"y-axis [Mpc]",rotation="vertical",fontsize=12)
    ax[0].xaxis.tick_top()
    ax[0].set_xlabel('Frequency [MHz]',fontsize=12)    
    ax[0].xaxis.set_label_position('top')
    ax[x].set_xlabel("Redshift",fontsize=12)
    plt.tight_layout()
    for y in range(len(parameters)):
        pos1=ax[x-y].get_position().get_points()+[[0.02,0.08/len(parameters)-0.02],[0.02,0.08/len(parameters)-0.02]]
        pos2=ax[y].get_position().get_points()+[[0.02,0.08/len(parameters)-0.03],[0.02,0.08/len(parameters)-0.03]]
        ax[x-y].set_position(Bbox(pos1-[[0,0.018],[0,0.018]]))
        ax[y].set_position(Bbox(pos2+[[0,0.018],[0,0.018]]))
        ax[y].text(10,150,"Sim",color="w",fontsize=14)
        ax[x-y].text(10,150,"Opt Mock",color="w",fontsize=14)

    # Use a colorbar with the "EoR" cmap from 21cmFAST
    cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=-150,vmax=30), cmap="EoR"), ax=ax,aspect=10*len(parameters))   
    cbar_label = r"$\delta T_B$ [mK]"
    cbar.ax.set_ylabel(cbar_label,fontsize=12)
    cbar.ax.tick_params(labelsize=12)
    os.makedirs(os.path.dirname(filename),exist_ok=True)
    plt.savefig(filename)
    plt.close()

def create_saliency_maps(filename,sim_lightcones,sim_model,mock_lightcones,mock_model,parameters,OMm):
    sim_saliency_maps=False
    mock_saliency_maps=False
    sim_saliency = Saliency(sim_model,
                           model_modifier=model_modifier,
                           clone=True)
    mock_saliency = Saliency(mock_model,
                            model_modifier=model_modifier,
                            clone=True)

    # Generate saliency maps for the requested parameters
    for para in parameters:
        def loss(output):
            return output[0][para]
        combined_sim_saliency=np.zeros((140,2350))
        combined_mock_saliency=np.zeros((140,2350))
        for lc in sim_lightcones:
            combined_sim_saliency+=sim_saliency(loss, lc.reshape(140,140,2350,1))[0][70]
        for lc in mock_lightcones:
            combined_mock_saliency+=mock_saliency(loss, lc.reshape(140,140,2350,1))[0][70]
        if sim_saliency_maps is False:
            sim_saliency_maps=np.array([combined_sim_saliency])
            mock_saliency_maps=np.array([combined_mock_saliency])
        else:
            sim_saliency_maps = np.append(sim_saliency_maps,np.array([combined_sim_saliency]),axis=0)
            mock_saliency_maps = np.append(mock_saliency_maps,np.array([combined_mock_saliency]),axis=0)
    saliency_maps=np.append(sim_saliency_maps,mock_saliency_maps,axis=0)
    
    # Define the light-cones as instances of the 21cmFAST LightCone class to use the plotting functions from 21cmFAST
    cosmo_params = p21c.CosmoParams(OMm=OMm)
    astro_params = p21c.AstroParams(INHOMO_RECO=True)
    user_params = p21c.UserParams(HII_DIM=140, BOX_LEN=200)
    flag_options = p21c.FlagOptions()
    sim_lightcone = p21c.LightCone(5.,user_params,cosmo_params,astro_params,flag_options,0,{"brightness_temp":sim_lightcones[0].astype(np.float32)},35.05)
    mock_lightcone = p21c.LightCone(5.,user_params,cosmo_params,astro_params,flag_options,0,{"brightness_temp":mock_lightcones[0].astype(np.float32)},35.05)
    plot(filename,sim_lightcone,mock_lightcone,parameters=parameters,saliency_maps=saliency_maps)
  
if __name__=="__main__":
    sim_model_file="../paper_results/3d_sim_6par/models/3D_21cmPIE_Net"
    mock_model_file="../paper_results/3d_optmock_6par/models/3D_21cmPIE_Net"
    sim_data_file="input/bare_sim.tfrecord"
    mock_data_file="input/optmocks.tfrecord"

    sim_model = keras.models.load_model(sim_model_file)
    mock_model = keras.models.load_model(mock_model_file)

    # Load data from a tfrecord file
    sim_dataset = tf.data.TFRecordDataset(sim_data_file)
    mock_dataset = tf.data.TFRecordDataset(mock_data_file)
    sim_dataset = sim_dataset.map(parse_function,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    mock_dataset = mock_dataset.map(parse_function,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    sim_data = np.array(list(sim_dataset.as_numpy_iterator()))
    mock_data = np.array(list(mock_dataset.as_numpy_iterator()))
    
    # Here Omega_m should be equal for all light-cones.
    OMm=sim_data[0][1][1]
    # Calculate and plot saliency maps for the requested parameters for the provided 3D CNNs which were trained on bare simulations and opt mocks respectively.
    # The saliency for each light-cone in sim_data and for each light-cone in mock_data will be stacked to reduce effects from local fluctuations.
    # Therefore all light-cones in sim_data and all light-cones in mock_data should be created using the same parameters.
    create_saliency_maps("output/saliency_maps/dm_saliency.png",sim_data[:,0],sim_model,mock_data[:,0],mock_model,parameters=[0,1],OMm=OMm) # Figure 5
    create_saliency_maps("output/saliency_maps/astro_saliency.png",sim_data[:,0],sim_model,mock_data[:,0],mock_model,parameters=[2,3,4,5],OMm=OMm) # Figure C1
