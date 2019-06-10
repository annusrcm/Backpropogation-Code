import numpy as np
import pandas as pd

def convert_to_list(x):
    k = []
    k.append(x)
    return k

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Input datasets
out_file = "data.csv"
df = pd.read_csv(open(out_file,'r'))
inputs = df[['X','Y','Z']]
inputs = inputs.values
e = np.array(df['output'])
expected_output = []
for i in e:
    expected_output.append([i])

# it converges at 100000 epoch
epochs = 10000
lr = 0.1
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 3, 3, 1

# Random weights and bias initialization
hidden_weights = np.random.uniform(size=(inputLayerNeurons, hiddenLayerNeurons))
hidden_bias = np.random.uniform(size=(1, hiddenLayerNeurons))
output_weights = np.random.uniform(size=(hiddenLayerNeurons, outputLayerNeurons))
output_bias = np.random.uniform(size=(1, outputLayerNeurons))

# print("Initial hidden weights:\n {}".format(hidden_weights))
# print("Initial hidden biases:\n {}".format(hidden_bias))
# print("Initial output weights:\n {}".format(output_weights))
# print("Initial output biases:\n {}".format(output_bias))


# Training algorithm
for _ in range(epochs):
    # Forward Propagation
    hidden_layer_activation = np.dot(inputs, hidden_weights)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output, output_weights)
    output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)

    # Backpropagation
    error = expected_output - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Updating Weights and Biases
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
    hidden_weights += inputs.T.dot(d_hidden_layer) * lr
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr

# print("Final hidden weights:\n {}".format(hidden_weights))
# print("Final hidden bias:\n {}".format(hidden_bias))
# print("Final output weights:\n {}".format(output_weights))
# print("Final output bias:\n {}".format(output_bias))


print("Output from neural network after {} epochs:\n {}".format(epochs, predicted_output))
