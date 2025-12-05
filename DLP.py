import numpy as np
import pandas as pd

housing_data = pd.read_csv("housing.csv")
housing_data = housing_data.fillna(housing_data.median(numeric_only=True))

temp=housing_data.drop(columns=['median_house_value'])
feature_matrix = pd.get_dummies(temp, columns=['ocean_proximity']).values

target_values = housing_data['median_house_value'].values.reshape(-1, 1)

feature_matrix = np.array(feature_matrix, dtype=np.float64)
target_values = np.array(target_values, dtype=np.float64)

feature_mean = feature_matrix.mean(axis=0, keepdims=True)
feature_std = feature_matrix.std(axis=0, keepdims=True)

target_mean = target_values.mean(axis=0, keepdims=True)
target_std = target_values.std(axis=0, keepdims=True)

if np.any(target_std == 0):
    for i in range(target_std.shape[0]):
        for j in range(target_std.shape[1]):
            if target_std[i, j] == 0:
                target_std[i, j] = 1

feature_matrix = (feature_matrix - feature_mean) / feature_std
target_values = (target_values - target_mean) / target_std

np.random.seed(10)
total_samples = feature_matrix.shape[0]
shuffled_indices = np.random.permutation(total_samples)
train_sample_size = int(0.7 * total_samples)
train_indices = shuffled_indices[:train_sample_size]
test_indices =  shuffled_indices[train_sample_size:]

training_features = feature_matrix[train_indices]
testing_features = feature_matrix[test_indices]
training_targets = target_values[train_indices]
testing_targets = target_values[test_indices]

target_mean = training_targets.mean()
target_std = training_targets.std()

print("Training Features shape:", training_features.shape)
print("Training Targets shape:", training_targets.shape)


def take_input():
    selected_activation_function = input("Choose activation function (relu/sigmoid): ").strip().lower()

    choice = input("What do you want to enter:\n 1).error \n2).Epochs  ").strip().lower()
    
    if choice == 1:
        stop_error = float(input("Enter error threshold: "))
        max_epochs = 500
    else:
        max_epochs = int(input("Enter max epochs: "))
        stop_error= None
    return selected_activation_function,max_epochs,stop_error

class NeuralNetwork:
    def __init__(self, input_size, hidden_layer_size, output_size, activation_function, learning_rate=0.001):
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.weights_input_hidden = np.random.uniform(-0.1, 0.1, (input_size, hidden_layer_size))
        self.weights_hidden_hidden = np.random.uniform(-0.1, 0.1, (hidden_layer_size, hidden_layer_size))
        self.weights_hidden_output = np.random.uniform(-0.1, 0.1, (hidden_layer_size, output_size))
        
        self.bias_hidden_1 = np.zeros((1, hidden_layer_size))
        self.bias_hidden_2 = np.zeros((1, hidden_layer_size))
        self.bias_output = np.zeros((1, output_size))

        self.activation_function = activation_function

    def activate(self, values):
        if self.activation_function == "sigmoid":
            return 1 / (1 + np.exp(-values))
        elif self.activation_function == "relu":
            return np.maximum(0, values)

    def activate_derivative(self, values):
        if self.activation_function == "relu":
            return (values > 0).astype(float)

        elif self.activation_function == "sigmoid":
            sigmoid_values = 1 / (1 + np.exp(-values))
            return sigmoid_values * (1 - sigmoid_values)
        else:
            return np.ones_like(values)

    def forward(self, input_data):
        self.hidden_layer_1_input = np.dot(input_data, self.weights_input_hidden) + self.bias_hidden_1
        self.hidden_layer_1_output = self.activate(self.hidden_layer_1_input)

        self.hidden_layer_2_input = np.dot(self.hidden_layer_1_output, self.weights_hidden_hidden) + self.bias_hidden_2
        self.hidden_layer_2_output = self.activate(self.hidden_layer_2_input)

        self.output_layer_input = np.dot(self.hidden_layer_2_output, self.weights_hidden_output) + self.bias_output
        return self.output_layer_input

    def compute_loss(self, predicted_output, actual_output):
        return np.mean((predicted_output - actual_output) ** 2)

    def backward(self, input_data, actual_output, predicted_output):
        o_error = (predicted_output - actual_output)
        o_gradient_weights = np.dot(self.hidden_layer_2_output.T, o_error)
        o_bias_gradient = np.sum(o_error, axis=0, keepdims=True)

        h2_error = np.dot(o_error, self.weights_hidden_output.T)
        h2_gradient = h2_error * self.activate_derivative(self.hidden_layer_2_input)
        h2_gradient_weights = np.dot(self.hidden_layer_1_output.T, h2_gradient)
        h2_gradient_bias = np.sum(h2_gradient, axis=0, keepdims=True)

        h1_error = np.dot(h2_gradient, self.weights_hidden_hidden.T)
        h1_gradient = h1_error * self.activate_derivative(self.hidden_layer_1_input)
        h1_weights_gradient = np.dot(input_data.T, h1_gradient)
        h1_bias = np.sum(h1_gradient, axis=0, keepdims=True)

        self.weights_hidden_output -= self.learning_rate * o_gradient_weights
        self.bias_output -= self.learning_rate * o_bias_gradient
        self.weights_hidden_hidden -= self.learning_rate * h2_gradient_weights
        self.bias_hidden_2 -= self.learning_rate * h2_gradient_bias
        self.weights_input_hidden -= self.learning_rate * h1_weights_gradient
        self.bias_hidden_1 -= self.learning_rate * h1_bias

    def train(self, train_features, train_targets, test_features, test_targets, epochs=100, batch_size=32, stop_loss_threshold=None):
        training_samples_count = train_features.shape[0]

        for epoch in range(epochs):
            shuffled_indices = np.random.permutation(training_samples_count)
            train_features = train_features[shuffled_indices]
            train_targets = train_targets[shuffled_indices]
            batch_num = 1
            for i in range(0, training_samples_count, batch_size):
                batch_features = train_features[i:i+batch_size]
                batch_targets = train_targets[i:i+batch_size]
                batch_predictions = self.forward(batch_features)
                self.backward(batch_features, batch_targets, batch_predictions)
                
                batch_loss = self.compute_loss(batch_predictions, batch_targets)
                print(f"Epoch: {epoch + 1}, Batch: {batch_num}, Loss: {batch_loss:.4f}")
                batch_num += 1
            fTrain = self.forward(train_features)
            fTest = self.forward(test_features)
            train_loss = self.compute_loss(fTrain, train_targets)
            test_loss = self.compute_loss(fTest, test_targets)
            print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}\n")


selected_activation_function,max_epochs,stop_error=take_input()

nn = NeuralNetwork(input_size=training_features.shape[1], hidden_layer_size=4, output_size=1,
                   activation_function=selected_activation_function, learning_rate=0.01)

nn.train(training_features, training_targets, testing_features, testing_targets, epochs=max_epochs, batch_size=50, stop_loss_threshold=stop_error)
