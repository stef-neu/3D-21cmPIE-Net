import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Model():
    def __init__(self,shape):
        self.inputs = keras.Input(shape=shape)

    def buildModel(self):
        # Convolution and pooling layers
        x = layers.Conv3D(filters=32, kernel_size=(3,3,102),strides=(1,1,102), activation="relu")(self.inputs)
        x = layers.Conv3D(filters=32, kernel_size=(3,3,2), activation="relu")(x)
        x = layers.MaxPooling3D(pool_size=(2,2,1))(x)
        x = layers.Conv3D(filters=64, kernel_size=(3,3,2), activation="relu")(x)
        x = layers.ZeroPadding3D(padding=(1,1,0))(x)
        x = layers.Conv3D(filters=64, kernel_size=(3,3,2), activation="relu")(x)
        x = layers.MaxPooling3D(pool_size=(2,2,1))(x)
        x = layers.Conv3D(filters=128, kernel_size=(3,3,2), activation="relu")(x)
        x = layers.ZeroPadding3D(padding=(1,1,0))(x)
        x = layers.Conv3D(filters=128, kernel_size=(3,3,2), activation="relu")(x)

        # Global average pooling and dense layers
        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        outputs = layers.Dense(6, activation="sigmoid")(x)

        # Defining and compiling our model
        self.model = keras.Model(inputs=self.inputs, outputs=outputs, name="3D_21cmPIE-Net")
        self.model.summary()     
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=4e-4,epsilon=1e-7,amsgrad=True),
                           loss=keras.losses.MeanSquaredError())
        return self.model

    def fitModel(self, training_data, test_data, **kwargs):
        self.model.fit(training_data,**kwargs)
        print("Evaluating test scores")
        test_scores = self.model.evaluate(test_data)
        print("Test loss: " + str(test_scores))

    def saveModel(self, location):
        self.model.save(location)
        return self.model
        
    def loadModel(self, location):
        self.model = keras.models.load_model(location)
        self.model.summary()
        return self.model
