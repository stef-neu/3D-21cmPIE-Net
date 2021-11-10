import tensorflow as tf
from tensorflow import keras
import ReadData, Model, Plotting
import optparse, sys

o = optparse.OptionParser()
o.set_usage('runCNN.py [options] [NN directory]')

o.add_option('--astroOnly', dest='ao', default=False, action="store_true",
             help="If true, assumes an astro-only dataset with 4 labels per light-cone")

o.add_option('--data', dest='data', default="../simulations/output/*.tfrecord",
             help="File pattern for the light-cone files")

opts, args = o.parse_args(sys.argv[1:])

if __name__ == "__main__":
    # Read in the dataset
    height_dim = 140 # spatial pixels 
    lc_dim = 2350 # pixel in frequency direction
    rd=ReadData.ReadData(x=height_dim,y=height_dim,z=lc_dim,astro_only=opts.ao)
    rd.read(opts.data,test_only=True)
    test_ds=rd.prepare_for_training(batch_size=8,cache=False,test_only=True)

    # Create a new neural network model or load a pretrained one
    model_handler=Model.Model(shape=(height_dim,height_dim,lc_dim,1))
    model=model_handler.load_model(args[0])

    # Evaluate the provided test-set
    model_handler.eval_model(test_ds)
    
    # Calculate R^2 values and create scatter plots.
    plot=Plotting.Plotting(model,test_ds,astro_only=opts.ao)
    plot.calculate_r2()
    plot.plot()
