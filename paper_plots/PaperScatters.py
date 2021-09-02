import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import functools, os

def plot(filename,origin,cbar=False,axis_hists=False,inlet_diagram=False,vmin=0,vmax=0,vmax_outer=0,inlet_scatter=False):
    # Load npz file with test labels and predictions from a neural network
    with np.load(origin) as data:
        test_labels = data["test_labels"]
        test_pred = data["test_pred"]

    # Additional test labels and predictions for opt and mod mocks. Only required for Figure 2
    if inlet_diagram:
        with np.load("../paper_results/3DOptMock6Par/TestValues.npz") as data:
            labels2 = data["test_labels"]
            pred2 = data["test_pred"]

        with np.load("../paper_results/3DModMock6Par/TestValues.npz") as data:
            labels3 = data["test_labels"]
            pred3 = data["test_pred"]

    # Define the name, parameter range and latex code for each parameter
    parameters=[["WDM",0.3,10,"$m_{WDM}$"],["OMm",0.2,0.4,"$\Omega_m$"],["LX",38,42,"$L_X$"],["E0",100,1500,"$E_0$"],["Tvir",4,5.3,"$T_{vir}$"],["Zeta",10,250,"$\zeta$"]]

    # Revert parameter normalizations to the [0,1] range
    for para in range(6):
        test_pred[:,para]=[x*(parameters[para][2]-parameters[para][1])+parameters[para][1] for x in test_pred[:,para]]
        test_labels[:,para]=[x*(parameters[para][2]-parameters[para][1])+parameters[para][1] for x in test_labels[:,para]]
        if inlet_diagram:
            pred2[:,para]=[x*(parameters[para][2]-parameters[para][1])+parameters[para][1] for x in pred2[:,para]]
            labels2[:,para]=[x*(parameters[para][2]-parameters[para][1])+parameters[para][1] for x in labels2[:,para]]

    # Create plots for each parameter
    for para in range(6):
        if inlet_scatter and inlet_scatter[0]!=parameters[para][0]:
            continue
        # Define axis limits
        limlow=parameters[para][1]-(parameters[para][2]-parameters[para][1])*0.05
        limhigh=parameters[para][2]+(parameters[para][2]-parameters[para][1])*0.05

        # Include y_true-y_pred at y_pred and y_pred-y_true as axis plots
        if axis_hists:
            fig = plt.figure(figsize=((6 if cbar else 5.1),5))
            # definitions for the axes to make everything look nice
            left, width = (0.125 if cbar else 0.135), (0.55 if cbar else 0.625)
            bottom, height = 0.11, (0.62 if cbar else 0.60)
            spacing = 0.03
            outwidth = 0.2
            rect_scatter = [left, bottom, width, height]
            rect_histx = [left, bottom + height + spacing, width, (0.18 if cbar else outwidth)]
            rect_histy = [left + width + spacing, bottom, outwidth, height]
            ax = fig.add_axes(rect_scatter)
            # Create the main plot and add axis histograms
            ax.hist2d(test_labels[:,para],test_pred[:,para],bins=50,range=[[limlow,limhigh],[limlow,limhigh]],cmap='plasma',vmin=vmin,vmax=(vmax if vmax else None))
            ax_histx = fig.add_axes(rect_histx, sharex=ax)
            ax_histy = fig.add_axes(rect_histy, sharey=ax)

            # Plot the x-axis plot
            limit=(np.amin(test_pred[:,para]-test_labels[:,para]),np.amax(test_pred[:,para]-test_labels[:,para]))
            limit=(-limit[0] if limit[1]+limit[0]<0 else limit[1])
            hist,xedges,yedges,image=ax_histx.hist2d(test_labels[:,para], test_pred[:,para]-test_labels[:,para], bins=(20,20),cmap="plasma",range=[[limlow,limhigh],[-limit,limit]],vmin=vmin,vmax=(vmax_outer if vmax_outer else None))

            # And one sigma range as errorbars
            mean=np.array([])
            variance=np.array([])
            start=0
            end=0
            for edge in range(len(xedges)-1):
                values=np.array([])
                for x,y in enumerate(test_labels[:,para]):
                    if xedges[edge]<y<xedges[edge+1]:
                        values=np.append(values,test_pred[x,para]-test_labels[x,para])
                if len(values):
                    mean=np.append(mean,np.array([np.mean(values)]))
                    var=np.sqrt(np.sum((values-mean[edge-start])**2)/len(values))
                    variance=np.append(variance,var)
                elif len(mean):
                    end+=1
                else:
                    start+=1
            ax_histx.errorbar((xedges[start+1:len(xedges)-end]+xedges[start:-1-end])/2,mean,yerr=variance,color="lightgrey",fmt=".")

            # Plot the y-axis plot
            limit=(np.amin(test_labels[:,para]-test_pred[:,para]),np.amax(test_labels[:,para]-test_pred[:,para]))
            limit=(-limit[0] if limit[1]+limit[0]<0 else limit[1])
            hist,xedges,yedges,image=ax_histy.hist2d(test_labels[:,para]-test_pred[:,para], test_pred[:,para], bins=(20,20), cmap="plasma",range=[[-limit,limit],[limlow,limhigh]],vmin=vmin,vmax=(vmax_outer if vmax_outer else None))

            # And one sigma range as errorbars
            mean=np.array([])
            variance=np.array([])
            start=0
            end=0
            for edge in range(len(yedges)-1):
                values=np.array([])
                for x,y in enumerate(test_pred[:,para]):
                    if yedges[edge]<y<yedges[edge+1]:
                        values=np.append(values,test_labels[x,para]-test_pred[x,para])
                if len(values):
                    mean=np.append(mean,np.array([np.mean(values)]))
                    var=np.sqrt(np.sum((values-mean[edge-start])**2)/len(values))
                    variance=np.append(variance,var)
                elif len(mean):
                    end+=1
                else:
                    start+=1
            ax_histy.errorbar(mean,(yedges[start+1:len(yedges)-end]+yedges[start:-1-end])/2,xerr=variance,color="lightgrey",fmt=".")
            if para==1: # Fine tuning to make the numbers more visible
                ax_histx.set_yticks(ticks=[-0.04,0,0.04])
                ax_histy.set_xticks(ticks=[-0.04,0,0.04])
            if cbar:
                cb = fig.colorbar(image,ax=ax_histy,aspect=22)
                cb.set_label('Number of entries')
            ax_histx.tick_params(axis="x", labelbottom=False)
            ax_histy.tick_params(axis="y", labelleft=False)
            ax_histx.set_title(parameters[para][3])

        # No axis hists
        else:
            # Create main plot
            fig,ax=plt.subplots(figsize=((5.4,5) if not cbar else None))
            plt.hist2d(test_labels[:,para],test_pred[:,para],bins=50,range=[[limlow,limhigh],[limlow,limhigh]],cmap='plasma',vmin=vmin,vmax=(vmax if vmax else None))
            if cbar:
                cb = plt.colorbar()
                cb.set_label('Number of entries')

            # Show pred-label distributions for simulations, opt mocks and mod mocks. The distributions are shown in an inserted histhogram.
            if inlet_diagram:
                axins = inset_axes(ax, width="35%", height="35%", loc=4, borderpad=2)
                axins.hist((test_pred[:,para]-test_labels[:,para])/(parameters[para][2]-parameters[para][1]),bins=51,label="Sim",alpha=1,edgecolor="red",histtype="step")
                axins.hist((pred2[:,para]-labels2[:,para])/(parameters[para][2]-parameters[para][1]),bins=51,label="Mock",alpha=1,edgecolor="blue",histtype="step")
                axins.hist(pred3[:,para]-labels3[:,para],bins=51,label="ModMock",alpha=1,edgecolor="green",histtype="step")
                axins.set_xlim(-0.5,0.5)
                axins.xaxis.label.set_color('white')
                axins.yaxis.label.set_color('white')
                axins.tick_params(colors='white')
                axins.set_title("Pred-Label",color="white",fontsize=10)
                axins.get_yaxis().set_visible(False)

        # Insert a smaller histogram of a different parameter into the main hist to save space
        para2=False
        if inlet_scatter and inlet_scatter[0]==parameters[para][0]:
            for index,element in enumerate(parameters):
                if inlet_scatter[1]==element[0]:
                    para2=index
            axins = inset_axes(ax, width="35%", height="35%", loc=2, borderpad=2)
            for spine in axins.spines.values():
                spine.set_edgecolor('white')
            axins.hist2d(test_labels[:,para2],test_pred[:,para2],bins=20,range=[[inlet_scatter[2],inlet_scatter[3]],[inlet_scatter[2],inlet_scatter[3]]],cmap='plasma',vmin=vmin,vmax=(vmax if vmax else None))
            axins.xaxis.label.set_color('white')
            axins.yaxis.label.set_color('white')
            axins.tick_params(colors='white')
            axins.set_title(parameters[para2][3],color="white",fontsize=10)

        if not axis_hists:
            ax.set_title(parameters[para][3])
        ax.set_xlabel('Labels')
        ax.set_ylabel('Predictions')
        ax.grid(True)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename+parameters[para][0]+(parameters[para2][0] if para2 else "")+".png")
        plt.close()

if __name__=="__main__":
    # Create the paper scatter plots
    plot("output/ScatterPlots/CompDiagrams_","../paper_results/3DSim6Par/TestValues.npz",cbar=True,inlet_diagram=True,vmax=17) # Figure 2
    plot("output/ScatterPlots/AxisHists_","../paper_results/3DSim6Par/TestValues.npz",cbar=True,axis_hists=True,vmax=17,vmax_outer=25) # Figure 3
    plot("output/ScatterPlots/SimMockTransfer_","../paper_results/TransferSimCNNMockData/TestValues.npz",cbar=True,vmax=15) # Figure B1, right
    plot("output/ScatterPlots/MockSimTransfer_","../paper_results/TransferMockCNNSimData/TestValues.npz",cbar=False,vmax=15) # Figure B1, left
    plot("output/ScatterPlots/SimMockTransfer_","../paper_results/TransferSimCNNMockData/TestValues.npz",cbar=True,vmax=15,inlet_scatter=["OMm","WDM",0.3,4]) # Figure 4, right
    plot("output/ScatterPlots/MockSimTransfer_","../paper_results/TransferMockCNNSimData/TestValues.npz",cbar=False,vmax=15,inlet_scatter=["OMm","WDM",0.3,4]) # Figure 4, left
    
