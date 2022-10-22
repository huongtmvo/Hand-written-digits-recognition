from PIL import Image
import numpy as np 
from tensorflow import keras
import sys

def process_image(path, model_type):
    size = 28
    im = Image.open(path).resize((size,size)).convert("L")
    im = np.array(im)
    if model_type == 'mlp':
        im = im.reshape(1, size*size)
    else:
        im = im.reshape(1, size, size, 1)
    im = im.astype('float32') / 255
    print(im.shape)
    return im 

if __name__ == '__main__':
    model_type = sys.argv[1]
    if model_type == 'mlp':
        model = keras.models.load_model('./models/mlp')
    else:
        model = keras.models.load_model('./models/cnn')
    print(f"loading model ...")
    print(f"model's summary:")
    model.summary()
    path = sys.argv[2]
    print(f"path = {path}")
    im = process_image(path, model_type)
    out = model.predict(im)
    print(f"out = {out}")
    print(f"Model predicted output = {np.argmax(out)}")