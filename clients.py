import tensorflow as tf
import cv2

class User:
    def __init__(self, local_data, local_label, is_to_preprocess):
        self.dataset = local_data
        self.label = local_label
        self.is_to_preprocess = is_to_preprocess

        self.dataset_size = local_data.shape[0]
        self._index_in_train_epoch = 0

        if self.is_to_preprocess:
            self.preprocess()

    def next_batch(self, batch_size):
        start = self._index_in_train_epoch
        self._index_in_train_epoch += batch_size
        if self._index_in_train_epoch > self.dataset_size:
            indices = np.arange(self.dataset_size)
            np.random.shuffle(indices)
            self.dataset = self.dataset[indices]
            self.label = self.label[indices]
            if self.is_to_preprocess:
                self.preprocess()
            start = 0
            self._index_in_train_epoch = batch_size
        end = self._index_in_train_epoch
        return self.dataset[start:end], self.label[start:end]

    def preprocess(self):
        # Utilize TensorFlow's image preprocessing utilities for efficiency
        new_images = tf.image.random_flip_left_right(self.dataset)
        new_images = tf.image.random_flip_up_down(new_images)
        new_images = tf.image.random_crop(new_images, size=[self.dataset_size, 24, 24, 3])
        new_images = tf.image.per_image_standardization(new_images)
        self.dataset = new_images.numpy()

class Clients:
    def __init__(self, num_of_clients, dataset_name, batch_size, epoch, sess, train_op, inputs, labels, is_iid):
        self.num_of_clients = num_of_clients
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.epoch = epoch
        self.session = sess
        self.train_op = train_op
        self.inputs = inputs
        self.labels = labels
        self.is_iid = is_iid
        self.clients_set = {}

        self.dataset_balance_allocation()

    def dataset_balance_allocation(self):
        dataset = DataSet(self.dataset_name, self.is_iid)
        local_data_size = dataset.train_data_size // self.num_of_clients

        for i in range(self.num_of_clients):
            start = i * local_data_size
            end = start + local_data_size
            client_data = dataset.train_data[start:end]
            client_labels = dataset.train_label[start:end]
            preprocess = 1 if self.dataset_name == 'cifar10' else 0
            self.clients_set[f'client{i}'] = User(client_data, client_labels, preprocess)

    def client_update(self, client_name, global_vars):
        user = self.clients_set[client_name]
        for epoch in range(self.epoch):
            for _ in range(user.dataset_size // self.batch_size):
                batch_data, batch_labels = user.next_batch(self.batch_size)
                self.session.run(self.train_op, feed_dict={self.inputs: batch_data, self.labels: batch_labels})
        
        return self.session.run(tf.trainable_variables())
