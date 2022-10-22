from keras import models
from keras import layers 
from keras.datasets import mnist
from keras.utils import to_categorical
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def create_cnn_model():
    cnn = models.Sequential()
    cnn.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
    cnn.add(layers.MaxPooling2D(2,2))
    cnn.add(layers.Conv2D(64, (3,3), activation = 'relu'))
    cnn.add(layers.MaxPooling2D(2,2))
    cnn.add(layers.Conv2D(64, (3,3), activation='relu'))
    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(64, activation = 'relu'))
    cnn.add(layers.Dense(10, activation = 'softmax'))
    cnn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return cnn


def process_data():
    (train_data, train_label), (test_data, test_label) = mnist.load_data()
    train_data = train_data.reshape(60000, 28, 28, 1)
    test_data = test_data.reshape(10000, 28, 28, 1)
    # scale data in range [0,1]
    train_data = train_data.astype('float32') / 255
    test_data = test_data.astype('float32') / 255
    # convert label to one hot vector
    train_label = to_categorical(train_label)
    test_label = to_categorical(test_label)
    return train_data, train_label, test_data, test_label


if __name__ == '__main__':
    train_data, train_label, test_data, test_label = process_data()
    cnn = create_cnn_model()
    # training model 
    cnn.fit(train_data, train_label, epochs = 5, batch_size = 60)
    # testing model 
    test_loss, test_acc = cnn.evaluate(test_data, test_label)
    print(f"Test accuracy = {test_acc} and test loss = {test_loss}")
    cnn.save("./models/cnn")
