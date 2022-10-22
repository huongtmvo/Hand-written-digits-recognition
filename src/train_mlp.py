from keras import models
from keras import layers 
from keras.datasets import mnist
from keras.utils import to_categorical
# import tensorflowjs as tfjs
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def create_mlp_model():
    mlp = models.Sequential()
    mlp.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
    mlp.add(layers.Dense(10, activation='softmax'))
    mlp.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return mlp


def process_data():
    (train_data, train_label), (test_data, test_label) = mnist.load_data()
    train_data = train_data.reshape(60000, 28*28)
    test_data = test_data.reshape(10000,28*28)
    # scale data in range [0,1]
    train_data = train_data.astype('float32') / 255
    test_data = test_data.astype('float32') / 255
    # convert label to one hot vector
    train_label = to_categorical(train_label)
    test_label = to_categorical(test_label)
    return train_data, train_label, test_data, test_label


if __name__ == '__main__':
    train_data, train_label, test_data, test_label = process_data()
    mlp = create_mlp_model()
    # training model 
    mlp.fit(train_data, train_label, epochs = 5, batch_size = 120)
    # testing model 
    test_loss, test_acc = mlp.evaluate(test_data, test_label)
    print(f"Test accuracy = {test_acc} and test loss = {test_loss}")
    mlp.save("./models/mlp")

    # tfjs.converters.save_keras_model(mlp, 'mlp_tfjs')
