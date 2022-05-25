import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from urllib import request
from io import BytesIO
from waste import CLASSES

def idwaste(url: str, modelpath: str):
    model = tf.keras.models.load_model(modelpath)

    res = request.urlopen(url).read()
    img = Image.open(BytesIO(res)).resize((256,256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)

    return {
            "class": CLASSES[np.argmax(predictions)],
            "certainty": predictions[0][np.argmax(predictions)]*100
    }

