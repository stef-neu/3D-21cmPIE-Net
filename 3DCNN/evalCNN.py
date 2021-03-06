import tensorflow as tf
from tensorflow import keras
import ReadData, Model, Plotting
import optparse, sys

o = optparse.OptionParser()
o.set_usage('runCNN.py [options] [NN_directory]')

o.add_option('--astroOnly', dest='ao', default=False, action="store_true",
             help="If true, assumes an astro-only dataset with 4 labels per light-cone")

o.add_option('--meanAverage', dest='ma', default=False, action="store_true",
             help="Evaluate on a mean averaged dataset.")

o.add_option('--data', dest='data', default="../simulations/output/*.tfrecord",
             help="File pattern for the light-cone files")

opts, args = o.parse_args(sys.argv[1:])

if __name__ == "__main__":
    # Read in the dataset
    height_dim = 140 # spatial pixels 
    lc_dim = 2350 # pixel in frequency direction
    rd=ReadData.ReadData(x=height_dim,y=height_dim,z=lc_dim,astro_only=opts.ao,mean_average=opts.ma)
    rd.read(opts.data,test_only=True)
    test_ds=rd.prepare_for_training(batch_size=8,cache=False,test_only=True)

    # Create a new neural network model or load a pretrained one
    model_handler=Model.Model(shape=(height_dim,height_dim,lc_dim,1))
    if len(args)<1:
        model_directory="../paper_results/3d_sim_6par/models/3D_21cmPIE_Net"
        print('N_directory was not given, setting to default ../paper_results/3d_sim_6par/models/3D_21cmPIE_Net')
    else:
        model_directory=args[0]
    model=model_handler.load_model(model_directory)

    # Evaluate the provided test-set
    model_handler.eval_model(test_ds)
    
    # Calculate R^2 values and create scatter plots.
    plot=Plotting.Plotting(model,test_ds,astro_only=opts.ao)
    plot.calculate_r2()
    plot.plot()
