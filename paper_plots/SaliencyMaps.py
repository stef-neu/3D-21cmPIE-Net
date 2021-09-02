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

# Plot the saliency maps over the light-cones. Axis scales depend on the value of Omega_m. This function expects all lightcones to have the same Omega_m
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
        ax[x].text(10,10,"$\delta "+parameter_list[para][3][1:],color="w",fontsize=14)
        ax[x].set_ylabel("")
    fig.text(0.01,0.59+0.0275*len(parameters),"y-axis [Mpc]",rotation="vertical")
    fig.text(0.01,0.20+0.005*len(parameters),"y-axis [Mpc]",rotation="vertical")
    ax[0].xaxis.tick_top()
    ax[0].set_xlabel('Frequency [MHz]')    
    ax[0].xaxis.set_label_position('top')
    ax[x].set_xlabel("Redshift")
    plt.tight_layout()
    for y in range(len(parameters)):
        pos1=ax[x-y].get_position().get_points()+[[0.04,0.08/len(parameters)],[0.04,0.08/len(parameters)+0.02]]
        pos2=ax[y].get_position().get_points()+[[0.04,0.08/len(parameters)],[0.04,0.08/len(parameters)]]
        ax[x-y].set_position(Bbox(pos1-[[0,0.018],[0,0.018]]))
        ax[y].set_position(Bbox(pos2+[[0,0.018],[0,0.018]]))
        ax[y].text(10,150,"Sim",color="w",fontsize=14)
        ax[x-y].text(10,150,"Opt Mock",color="w",fontsize=14)

    # Use a colorbar with the "EoR" cmap from 21cmFAST
    cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=-150,vmax=30), cmap="EoR"), ax=ax,aspect=10*len(parameters))   
    cbar_label = r"$\delta T_B$ [mK]"
    cbar.ax.set_ylabel(cbar_label)
    os.makedirs(os.path.dirname(filename),exist_ok=True)
    plt.savefig(filename)
    plt.close()

def createSaliencyMaps(filename,sim_lightcones,sim_model,mock_lightcones,mock_model,parameters,OMm):
    simSaliency_maps=False
    mockSaliency_maps=False
    simSaliency = Saliency(sim_model,
                           model_modifier=model_modifier,
                           clone=True)
    mockSaliency = Saliency(mock_model,
                            model_modifier=model_modifier,
                            clone=True)

    # Generate saliency maps for the requested parameters
    for para in parameters:
        def loss(output):
            return output[0][para]
        combined_simSaliency=np.zeros((140,2350))
        combined_mockSaliency=np.zeros((140,2350))
        for lc in sim_lightcones:
            combined_simSaliency+=simSaliency(loss, lc.reshape(140,140,2350,1))[0][70]
        for lc in mock_lightcones:
            combined_mockSaliency+=mockSaliency(loss, lc.reshape(140,140,2350,1))[0][70]
        if simSaliency_maps is False:
            simSaliency_maps=np.array([combined_simSaliency])
            mockSaliency_maps=np.array([combined_mockSaliency])
        else:
            simSaliency_maps = np.append(simSaliency_maps,np.array([combined_simSaliency]),axis=0)
            mockSaliency_maps = np.append(mockSaliency_maps,np.array([combined_mockSaliency]),axis=0)
    saliency_maps=np.append(simSaliency_maps,mockSaliency_maps,axis=0)
    
    # Define the light-cones as instances of the 21cmFAST LightCone class to use the plotting functions from 21cmFAST
    cosmo_params = p21c.CosmoParams(OMm=OMm)
    astro_params = p21c.AstroParams(INHOMO_RECO=True)
    user_params = p21c.UserParams(HII_DIM=140, BOX_LEN=200)
    flag_options = p21c.FlagOptions()
    simLightcone=p21c.LightCone(5.,user_params,cosmo_params,astro_params,flag_options,0,{"brightness_temp":sim_lightcones[0]},35.05)
    mockLightcone=p21c.LightCone(5.,user_params,cosmo_params,astro_params,flag_options,0,{"brightness_temp":mock_lightcones[0]},35.05)
    plot(filename,simLightcone,mockLightcone,parameters=parameters,saliency_maps=saliency_maps)
  
if __name__=="__main__":
    simModelFile="../paper_results/3DSim6Par/Models/3D_21cmPIE_Net"
    mockModelFile="../paper_results/3DOptMock6Par/Models/3D_21cmPIE_Net"
    simDataFile="../input/BareSim.tfrecord"
    mockDataFile="../input/optMock.tfrecord"

    simModel = keras.models.load_model(simModelFile)
    mockModel = keras.models.load_model(mockModelFile)

    # Load data from a tfrecord file
    simDataset = tf.data.TFRecordDataset(simDataFile)
    mockData = tf.data.TFRecordDataset(mockDataFile)
    simDataset = simDataset.map(parse_function,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    mockDataset = mockDataset.map(parse_function,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    simData = np.array(list(simDataset.as_numpy_iterator()))
    mockData = np.array(list(mockDataset.as_numpy_iterator()))
    
    # Here Omega_m should be equal for all lightcones.
    OMm=simData[1][1]

    # Calculate and plot saliency maps for the requested parameters for the provided 3D CNNs which were trained on bare simulations and opt mocks respectively. The saliency for each lightcone in simData and for each lightcone in mockData will be stacked to reduce effects from local fluctuations. Therefore all lightcones in simData and all lightcones in mockData should be created using the same parameters.
    # Parameters = m_WDM, Omega_m, L_X, E_0, T_vir, zeta
    createSaliencyMaps("output/SaliencyMaps/WDMOMmSaliency.png",simData,simModel,mockData,mockModel,parameters=[0,1],OMm=OMm) # Figure 5
    createSaliencyMaps("output/SaliencyMaps/AstroSaliency.png",simData,simModel,mockData,mockModel,parameters=[2,3,4,5],OMm=OMm) # Figure C1
