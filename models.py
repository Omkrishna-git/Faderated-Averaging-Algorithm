import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import Model

class Models(Model):
    def __init__(self, model_name):
        super(Models, self).__init__()
        self.model_name = model_name
        
        if self.model_name == 'mnist_2nn':
            self.build_mnist_2nn()
        elif self.model_name == 'mnist_cnn':
            self.build_mnist_cnn()
        elif self.model_name == 'cifar10_cnn':
            self.build_cifar10_cnn()
    
    def build_mnist_2nn(self):
        self.fc1 = Dense(200, activation='relu')
        self.fc2 = Dense(200, activation='relu')
        self.fc3 = Dense(10)  # No activation function here

    def build_mnist_cnn(self):
        self.conv1 = Conv2D(32, 5, activation='relu', padding='same')
        self.pool1 = MaxPooling2D(2)
        self.conv2 = Conv2D(64, 5, activation='relu', padding='same')
        self.pool2 = MaxPooling2D(2)
        self.flatten = Flatten()
        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(10)

    def build_cifar10_cnn(self):
        self.conv1 = Conv2D(64, 5, activation='relu', padding='same')
        self.pool1 = MaxPooling2D(2)
        self.conv2 = Conv2D(64, 5, activation='relu', padding='same')
        self.pool2 = MaxPooling2D(2)
        self.flatten = Flatten()
        self.fc1 = Dense(384, activation='relu')
        self.fc2 = Dense(192, activation='relu')
        self.fc3 = Dense(10)

    def call(self, inputs):
        if self.model_name in ['mnist_2nn']:
            x = self.fc1(inputs)
            x = self.fc2(x)
            return self.fc3(x)
        
        if self.model_name in ['mnist_cnn', 'cifar10_cnn']:
            x = self.conv1(inputs)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.fc2(x)
            return self.fc3(x)
