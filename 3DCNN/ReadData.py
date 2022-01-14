import tensorflow as tf
import numpy as np
import glob

class ReadData():
    def __init__(self, x,y,z,apply_filter=False, astro_only=False,mean_average=False):
        # Setup
        assert all(isinstance(i,int) for i in  [x,y,z]), "Please provide the light-cone coordinates as integers"
        self.height=x
        self.width=y
        self.img_length=z
        self.autotune=tf.data.experimental.AUTOTUNE
        self.apply_filter=apply_filter
        self.mean_average=mean_average
        if astro_only:
            self.paras=4
        else:
            self.paras=6
  
    # Function for parsing our tf.records files. We may apply our filters now if we haven't done so during the simulations
    def parse_function(self,files):
        keys_to_features = {"label":tf.io.FixedLenFeature((self.paras),tf.float32),
                            "image":tf.io.FixedLenFeature((),tf.string),
                            "tau":tf.io.FixedLenFeature((),tf.float32),
                            "gxH":tf.io.FixedLenFeature((92),tf.float32),
                            "z":tf.io.FixedLenFeature((92),tf.float32),
        }
        parsed_features = tf.io.parse_example(files, keys_to_features)
        image = tf.io.decode_raw(parsed_features["image"],tf.float16)
        image = tf.reshape(image,(self.height,self.width,self.img_length))
        if self.mean_average:
            mean = tf.math.reduce_mean(image,0,keepdims=True)
            mean = tf.math.reduce_mean(mean,1,keepdims=True)
            norm = tf.math.reduce_max(image-mean)-tf.math.reduce_min(image-mean)
        else:
            mean = 0.
            norm = 1250.
        #if not self.apply_filter or (parsed_features["gxH"][0]<0.1 and parsed_features["tau"]<0.089):
        if self.paras==6:
            return (image[0:self.height,0:self.width,0:self.img_length]-mean)/norm, tf.stack([(parsed_features["label"][0]-.3)/9.7,(parsed_features["label"][1]-.2)*5.,(parsed_features["label"][2]-38.)/4.,(parsed_features["label"][3]-100.)/1400.,(parsed_features["label"][4]-4.)/1.3,(parsed_features["label"][5]-10.)/240.],axis=-1) # m_WDM, Omega_m, L_X, E_0, T_vir, zeta
        elif self.paras==4:
            return (image[0:self.height,0:self.width,0:self.img_length]-mean)/norm, tf.stack([(parsed_features["label"][0]-38.)/4.,(parsed_features["label"][1]-100.)/1400.,(parsed_features["label"][2]-4.)/1.3,(parsed_features["label"][3]-10.)/240.],axis=-1) # L_X, E_0, T_vir, zeta
        else:
            return image[0:self.height,0:self.width,0:self.img_length], tf.stack([5.]*self.paras)
      
    def read(self,path,test_path=False,test_only=False):
        # Read in all files matching the given pattern.
        if test_only and test_path:
            paths = glob.glob(test_path)
        else:
            paths = glob.glob(path)
        np.random.shuffle(paths)
        print("Reading in "+str(len(paths))+" files.")
        ds = tf.data.TFRecordDataset(paths)
        ds = ds.map(self.parse_function,num_parallel_calls=self.autotune)
        if self.apply_filter:
            ds = ds.filter(lambda x,y: y[0]<=1)

        # To evaluate the NN on a test dataset, only the test dataset is required                          
        if test_only:
            self.test_ds=ds
            return self.test_ds
        
        # We may want to use a seperate test set
        elif test_path:
            test_paths=glob.glob(test_path)
            self.test_ds = tf.data.TFRecordDataset(test_paths)
            self.test_ds = self.test_ds.map(self.parse_function,num_parallel_calls=self.autotune)
            if self.apply_filter:
                self.test_ds = self.test_ds.filter(lambda x,y: y[0]<=1)
            self.train_ds=ds.window(8,9).flat_map(lambda *ds: tf.data.Dataset.zip(ds))
            self.vali_ds=ds.skip(8).window(1,9).flat_map(lambda *ds: tf.data.Dataset.zip(ds))

        # 80%/10%/10% split for training/validation/testing
        else:
            self.train_ds=ds.window(8,10).flat_map(lambda *ds: tf.data.Dataset.zip(ds))
            self.vali_ds=ds.skip(8).window(1,10).flat_map(lambda *ds: tf.data.Dataset.zip(ds))
            self.test_ds=ds.skip(9).window(1,10).flat_map(lambda *ds: tf.data.Dataset.zip(ds))

        return self.train_ds, self.vali_ds, self.test_ds

    
    def prepare_for_training(self, batch_size, cache=False,test_only=False):
        # Cache if dataset fits in memory or to cache it to specified files
        if cache:
            if isinstance(cache, str):
                # Dataset will be cached into the specified file
                if not test_only:
                    self.train_ds = self.train_ds.cache(cache) 
                    self.vali_ds = self.vali_ds.cache(cache)
                self.test_ds = self.test_ds.cache(cache)
            else:
                # Dataset will be cached in memory
                if not test_only:
                    self.train_ds = self.train_ds.cache()
                    self.vali_ds = self.vali_ds.cache()
                self.test_ds = self.test_ds.cache()

        if not test_only:
            self.train_ds = self.train_ds.batch(batch_size)
            self.vali_ds = self.vali_ds.batch(batch_size)
        self.test_ds = self.test_ds.batch(batch_size)

        if not test_only:
            self.train_ds = self.train_ds.prefetch(buffer_size=self.autotune)
            self.vali_ds = self.vali_ds.prefetch(buffer_size=self.autotune)
        self.test_ds = self.test_ds.prefetch(buffer_size=self.autotune)
        
        if test_only:
            return self.test_ds
        return self.train_ds, self.vali_ds, self.test_ds
