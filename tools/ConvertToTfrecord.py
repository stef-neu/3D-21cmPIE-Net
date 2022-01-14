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

# Converts npz file "path" to tfrecord file "filename"
def convert(path,filename):
    print("Converting "+path+" to "+filename)
    writer = tf.io.TFRecordWriter(filename)
    with np.load(path) as data:
        for x in range(len(data["image"])):
            lc_list = tf.train.BytesList(value=[data["image"][x].flatten().astype(np.float16).tobytes()])
            lb_list = tf.train.FloatList(value=data["label"][x])
            taulist = tf.train.FloatList(value=[data["tau"][x]])
            zlist = tf.train.FloatList(value=data["z"][x])
            gxHlist = tf.train.FloatList(value=data["gxH"][x])
            image = tf.train.Feature(bytes_list=lc_list)
            label = tf.train.Feature(float_list=lb_list)
            tau = tf.train.Feature(float_list=taulist)
            redshift = tf.train.Feature(float_list=zlist)
            gxH = tf.train.Feature(float_list=gxHlist)
            lc_dict={
                'image': image,
                'label': label,
                'tau': tau,
                'z': redshift,
                'gxH': gxH,
            }
            features = tf.train.Features(feature=lc_dict)
            labeled_data = tf.train.Example(features=features)
            writer.write(labeled_data.SerializeToString())
    writer.close()
 
if __name__ == "__main__":
    # All npz files matching the specified pattern are converted to tfrecord files 
    paths = glob.glob(opts.data)
    paths = [x for x in paths if ".npz" in x]
    for path in paths:
        if opts.td:
            convert(path,opts.td+path.split("/")[-1][:-9]+".tfrecord")
        else:
            convert(path,functools.reduce(lambda a,b:a+"."+b,path.split(".")[:-1])+".tfrecord")
