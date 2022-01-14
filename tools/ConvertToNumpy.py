import tensorflow as tf
import numpy as np
import glob, functools
import tensorflow_datasets as tfds
import optparse, sys

o = optparse.OptionParser()
o.set_usage('convertToNumpy.py [options]')

o.add_option('--astroOnly', dest='ao', default=False, action="store_true",
             help="If true, assumes an astro-only dataset with 4 labels per light-cone")

o.add_option('--targetDirectory', dest='td', default=False,
             help="Set a target directory for the converted files. By default the converted files will be stored in the original directory")

o.add_option('--data', dest='data', default="../simulations/output/*.tfrecord",
             help="File or pattern of files to convert. By default all files in simulations are converted")

opts, args = o.parse_args(sys.argv[1:])

if opts.ao:
    params=4
else:
    params=6

# Parse function to read tfrecord files.
def parse_function(files):
    keys_to_features = {"label":tf.io.FixedLenFeature((params),tf.float32),
                        "image":tf.io.FixedLenFeature((),tf.string),
                        "tau":tf.io.FixedLenFeature((1),tf.float32),
                        "gxH":tf.io.FixedLenFeature((92),tf.float32),
                        "z":tf.io.FixedLenFeature((92),tf.float32),}
    parsed_features = tf.io.parse_example(files, keys_to_features)
    image = tf.io.decode_raw(parsed_features["image"],tf.float16)
    image = tf.reshape(image,(140,140,2350))
    return image, tf.concat([parsed_features["label"],parsed_features["tau"],parsed_features["gxH"],parsed_features["z"]],axis=-1)

# Converts tfrecord file "path" to npz file "filename"
def convert(path,filename):
    print("Converting "+path+" to "+filename)
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(parse_function,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_numpy = tfds.as_numpy(dataset)
    images = False
    labels = np.array([])
    tau = np.array([])
    gxH = np.array([])
    z = np.array([])
    for ex in ds_numpy:
        if images is False:
            images=np.array([ex[0]])
            labels=np.array([ex[1][:params]])
            gxH=np.array([ex[1][params+1:params+93]])
            z=np.array([ex[1][params+93:]])
        else:
            images=np.append(images,np.array([ex[0]]),axis=0)
            labels=np.append(labels,np.array([ex[1][:params]]),axis=0)
            gxH=np.append(gxH,np.array([ex[1][params+1:params+93]]),axis=0)
            z=np.append(z,np.array([ex[1][params+93:]]),axis=0)
        tau=np.append(tau,np.array([ex[1][params:params+1]]))
    np.savez(filename,**{"label":labels,"image":images,"tau":tau,"gxH":gxH,"z":z})

if __name__ == "__main__":
    # All tfrecords files matching the specified pattern are converted to npz files
    paths = glob.glob(opts.data)
    paths = [x for x in paths if ".tfrecord" in x]
    for path in paths:
        if opts.td:
            convert(path,opts.td+path.split("/")[-1][:-9]+".npz")
        else:
            convert(path,functools.reduce(lambda a,b:a+"."+b,path.split(".")[:-1])+".npz")
