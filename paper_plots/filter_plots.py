import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == "__main__":
    # Load trained model
    model = keras.models.load_model("../paper_results/3d_optmock_6par/models/3D_21cmPIE_Net")
    # Plot each of the 32 filters from the first convolutional layer with weights averaged over the first two dimensions
    for y in range(32):
        weights = model.layers[1].weights[0].numpy()
        weights = weights.reshape(3,3,102,32)[:,:,:,y]
        weights = np.mean(weights,axis=(0,1))
        x = np.linspace(0,len(weights),len(weights))
        fig, ax = plt.subplots()
        ax.scatter(x,weights,s=4)
        ax.set_xlabel("Pixel in redshift direction",fontsize=18)
        ax.set_ylabel("Average Weight",fontsize=18)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.tight_layout()
        fig.autolayout = True
        os.makedirs("output/filter_plots/", exist_ok=True)
        plt.savefig("output/filter_plots/filter"+str(y)+".png") # 0,1,2,4,5,26 are in Figure C1
        plt.close()
