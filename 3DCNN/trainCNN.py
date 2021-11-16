import tensorflow as tf
from tensorflow import keras
import ReadData, Model, Plotting
import shutil, optparse, sys

o = optparse.OptionParser()
o.set_usage('trainCNN.py [options] [N_epochs]')

o.add_option('--astroOnly', dest='ao', default=False, action="store_true",
             help="If true, assumes an astro-only dataset with 4 labels per light-cone")

o.add_option('--meanAverage', dest='ma', default=False, action="store_true",
             help="Train and test on the mean averaged dataset.")

o.add_option('--data', dest='data', default="../simulations/output/*.tfrecord",
             help="File pattern for the light-cone files")

o.add_option('--continue', dest='cont', default=False, action="store_true",
             help="Uses the saved model file to continue the run. WARNING: This currently always uses the initial learning rate. Lower learning rates have to be set manually")

opts, args = o.parse_args(sys.argv[1:])

if __name__ == "__main__":
    if len(args)<1:
        N_epochs = 2
        print('N_epochs was not given, setting to default 2')
    else:
        N_epochs = int(args[0])
    # Read in the dataset
    height_dim = 140 # spatial pixels 
    lc_dim = 2350 # pixel in frequency direction
    # CH test: dims can be changed, load dims from lc?
    rd=ReadData.ReadData(x=height_dim,y=height_dim,z=lc_dim,astro_only=opts.ao,mean_average=opts.ma)
    rd.read(opts.data)
    train_ds, vali_ds, test_ds=rd.prepare_for_training(batch_size=8,cache=False)

    # Create a new neural network model or load a pretrained one
    model_handler=Model.Model(shape=(height_dim, height_dim, lc_dim, 1))
    if opts.cont:
        model=model_handler.load_model("output/models/3D_21cmPIE_Net")
    else:
        model=model_handler.build_model(n_parameters=(4 if opts.ao else 6))

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
    model_handler.fit_model(
        training_data=train_ds,
        test_data=test_ds,
        epochs=N_epochs,
        callbacks=[tensorboard_callback, early_stop_callback,plateau_callback],
        validation_data=vali_ds,
        )
    model=model_handler.save_model("output/models/3D_21cmPIE_Net")
    
    # Calculate R^2 values and create scatter plots.
    plot=Plotting.Plotting(model,test_ds,astro_only=opts.ao)
    plot.calculate_r2()
    plot.plot()
