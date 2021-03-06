import numpy as np
import matplotlib.pyplot as plt
import functools, os

class Plotting():
    def __init__(self,model,test_ds,astro_only=False,load_file=False):
        # Define parameter names, ranges and latex code
        if astro_only:
            self.parameters=[["LX",38,42,"$L_X$"],["E0",100,1500,"$E_0$"],["Tvir",4,5.3,"$T_{vir}$"],["Zeta",10,250,"$\zeta$"]]
        else:
            self.parameters=[["WDM",0.3,10,"$m_{WDM}$"],["OMm",0.2,0.4,"$\Omega_m$"],["LX",38,42,"$L_X$"],["E0",100,1500,"$E_0$"],["Tvir",4,5.3,"$T_{vir}$"],["Zeta",10,250,"$\zeta$"]]

        # Load saved test predictions and labels from a previous run
        if load_file:
            with np.load(load_file) as data:
                self.test_pred=data["test_pred"]
                self.test_labels=data["test_labels"]

        else:
            # Use the model to predict the values from the dataset
            self.test_pred = model.predict(test_ds)
            self.test_labels=False
            for image,label in test_ds:
                if self.test_labels is False:
                    self.test_labels = label.numpy()
                else:
                    self.test_labels = np.append(self.test_labels,label.numpy(),axis=0)

            # Save the test labels and predictions for further usage
            os.makedirs("output", exist_ok=True)
            np.savez("output/test_values.npz",**{"test_labels":self.test_labels, "test_pred":self.test_pred})

    def calculate_r2(self):
        for para in range(len(self.parameters)):
            # Calculate the R2-score for each parameter
            average=sum(self.test_labels[:,para])/len(self.test_labels[:,para])
            dividend=[(x-y)**2 for x, y in zip(self.test_labels[:,para],self.test_pred[:,para])]
            divisor=[(x-average)**2 for x in self.test_labels[:,para]]
            r2=1-functools.reduce(lambda a,b : a+b,dividend)/functools.reduce(lambda a,b : a+b,divisor)
            print("R²_{"+self.parameters[para][0]+"}=" + str(r2))

    def plot(self,path="output/scatter_plots"):
        os.makedirs(path, exist_ok=True)
        for para in range(len(self.parameters)):
            # Rescale the parameter values from [0,1] to their original scale
            self.test_pred[:,para]=[x*(self.parameters[para][2]-self.parameters[para][1])+self.parameters[para][1] for x in self.test_pred[:,para]]
            self.test_labels[:,para]=[x*(self.parameters[para][2]-self.parameters[para][1])+self.parameters[para][1] for x in self.test_labels[:,para]]
            fig,ax = plt.subplots()
            
            # Create scatter plots of true vs predicted values
            ax.scatter(self.test_labels[:,para],self.test_pred[:,para],s=10,color="r",label="test",alpha=0.1)
            ax.set_xlim(self.parameters[para][1]-(self.parameters[para][2]-self.parameters[para][1])*0.05,self.parameters[para][2]+(self.parameters[para][2]-self.parameters[para][1])*0.05)
            ax.set_ylim(self.parameters[para][1]-(self.parameters[para][2]-self.parameters[para][1])*0.05,self.parameters[para][2]+(self.parameters[para][2]-self.parameters[para][1])*0.05)
            ax.set_title(self.parameters[para][3])
            ax.set_xlabel('Labels')
            ax.set_ylabel('Predictions')
            ax.legend()
            ax.grid(True)
            plt.savefig(path+"/test_scatter_"+self.parameters[para][0]+".png")
            plt.close()

            # Create heatmap scatter plots of true vs predicted values
            plt.hist2d(self.test_labels[:,para],self.test_pred[:,para],bins=50,range=[[self.parameters[para][1]-(self.parameters[para][2]-self.parameters[para][1])*0.05,self.parameters[para][2]+(self.parameters[para][2]-self.parameters[para][1])*0.05],[self.parameters[para][1]-(self.parameters[para][2]-self.parameters[para][1])*0.05,self.parameters[para][2]+(self.parameters[para][2]-self.parameters[para][1])*0.05]],cmap='plasma')
            cb=plt.colorbar()
            cb.set_label('Number of entries')
            plt.title(self.parameters[para][3])
            plt.xlabel('Labels')
            plt.ylabel('Predictions')
            plt.tight_layout()
            fig.autolayout = True
            plt.savefig(path+"/test_heatmap_"+self.parameters[para][0]+".png")
            plt.close()
