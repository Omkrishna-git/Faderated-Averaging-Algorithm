import tensorflow as tf
from Models import Models
from clients import Clients
import os
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='GPU id to use')
parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='Number of clients')
parser.add_argument('-cf', '--cfraction', type=float, default=0.1, help='C fraction')
parser.add_argument('-E', '--epoch', type=int, default=5, help='Local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=10, help='Local train batch size')
parser.add_argument('-mn', '--modelname', type=str, default='mnist_2nn', help='Model name')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="Learning rate")
parser.add_argument('-vf', "--val_freq", type=int, default=5, help="Validation frequency")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='Save frequency')
parser.add_argument('-ncomm', '--num_comm', type=int, default=1000, help='Number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='Checkpoint save path')
parser.add_argument('-iid', '--IID', type=int, default=0, help='IID flag')

def test_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    test_mkdir(args.save_path)

    if args.modelname in ['mnist_2nn', 'mnist_cnn']:
        input_shape = [None, 784]
    elif args.modelname == 'cifar10_cnn':
        input_shape = [None, 24, 24, 3]

    model = Models(args.modelname)

    model.compile(optimizer=tf.keras.optimizers.SGD(args.learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    my_clients = Clients(args.num_of_clients, args.modelname, args.batchsize, args.epoch, 
                         model.train_op, model.inputs, model.labels, args.IID)

    for round_num in range(args.num_comm):
        print(f"Communication round {round_num}")
        selected_clients = np.random.choice(list(my_clients.clients_set.keys()), 
                                            size=int(max(args.num_of_clients * args.cfraction, 1)),
                                            replace=False)

        client_weights = []
        for client in selected_clients:
            client_weights.append(my_clients.client_update(client, model))

        # Aggregate client models
        global_weights = np.mean(client_weights, axis=0)
        model.set_weights(global_weights)

        if round_num % args.val_freq == 0:
            # Evaluate global model
            pass

        if round_num % args.save_freq == 0:
            model.save(os.path.join(args.save_path, f"model_round_{round_num}.h5"))
