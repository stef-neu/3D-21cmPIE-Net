import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == "__main__":
    # Load trained model
    model = keras.models.load_model("../paper_results/3DOptMock6Par/Models/3D_21cmPIE_Net")
    # Plot each of the 32 filters in the first convolutional layer with weights averaged over the first two dimensions
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
        os.makedirs("output/FilterPlots/", exist_ok=True)
        plt.savefig("output/FilterPlots/Filter"+str(y)+".png")
        plt.close()
