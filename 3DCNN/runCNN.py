import tensorflow as tf
from tensorflow import keras
import ReadData, Model, Plotting
import shutil, optparse, sys

o = optparse.OptionParser()
o.set_usage('calc_sense.py [options] [epochs]')

o.add_option('--astroOnly', dest='ao', default=False, action="store_true",
             help="If true, assumes an astro-only dataset with 4 labels per lightcone")

o.add_option('--data', dest='data', default="../simulations/output/FullPara/*.tfrecord",
             help="File pattern for the light-cone files")

o.add_option('--continue', dest='cont', default=False, action="store_true",
             help="Uses the saved model file to continue the run. WARNING: This currently always uses the initial learning rate. Lower learning rates have to be set manually")

opts, args = o.parse_args(sys.argv[1:])

if __name__ == "__main__":
    # Read in the dataset
    rd=ReadData.ReadData(x=140,y=140,z=2350)
    rd.read(opts.data)
    train_ds, vali_ds, test_ds=rd.prepare_for_training(batch_size=8,cache=False)

    # Create a new neural network model or load a pretrained one
    modelHandler=Model.Model(shape=(140, 140, 2350, 1))
    if opts.cont:
        model=modelHandler.loadModel("output/models/3D_21cmPIE_Net")
    else:
        model=modelHandler.buildModel()

    # Clear out any prior log data
    try:
        shutil.rmtree("logs")
    except:
        pass

    # Define callbacks
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir="logs/train_data/")
    plateau_callback = keras.callbacks.ReduceLROnPlateau(patience=5,cooldown=0,factor=0.5)
    early_stop_callback = keras.callbacks.EarlyStopping(monitor="val_loss",min_delta=0.0001,patience=10,restore_best_weights=True)

    # Run the neural network
    modelHandler.fitModel(
        training_data=train_ds,
        test_data=test_ds,
        epochs=int(args[0]),
        callbacks=[tensorboard_callback, early_stop_callback,plateau_callback],
        validation_data=vali_ds,
        )
    model=modelHandler.saveModel("output/models/3D_21cmPIE_Net")
    
    # Calculate R2 values and create scatter plots.
    plot=Plotting.Plotting(model,test_ds,astroOnly=opts.ao)
    plot.calculateR2()
    plot.plot()
