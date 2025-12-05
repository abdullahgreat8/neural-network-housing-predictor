# California Housing Price Predictor

A neural network implementation from scratch using NumPy to predict median house values in California districts based on various housing features.

## Project Description

This project implements a multi-layer neural network (MLP) from scratch without using deep learning frameworks like TensorFlow or PyTorch. The model predicts housing prices using the California Housing Dataset, featuring a two-hidden-layer architecture with customizable activation functions (ReLU or Sigmoid).

## Features

- **Custom Neural Network Implementation**: Built entirely with NumPy
- **Two Hidden Layers**: Flexible architecture with configurable hidden layer sizes
- **Multiple Activation Functions**: Support for ReLU and Sigmoid activation functions
- **Mini-batch Gradient Descent**: Efficient training with configurable batch sizes
- **Feature Engineering**: Automatic one-hot encoding for categorical variables
- **Data Preprocessing**: Z-score normalization for features and targets
- **Train-Test Split**: 70-30 split with reproducible random seed
- **Real-time Training Monitoring**: Displays batch-level and epoch-level losses

## Dataset

The project uses the California Housing Dataset (`housing.csv`), which includes:
- **longitude**: Geographic coordinate
- **latitude**: Geographic coordinate
- **housing_median_age**: Median age of houses in the district
- **total_rooms**: Total number of rooms
- **total_bedrooms**: Total number of bedrooms
- **population**: District population
- **households**: Number of households
- **median_income**: Median income of households
- **ocean_proximity**: Categorical feature (proximity to ocean)
- **median_house_value**: Target variable (house prices)

## Requirements

```
numpy
pandas
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd california-housing-price-predictor
```

2. Install required packages:
```bash
pip install numpy pandas
```

3. Ensure `housing.csv` is in the same directory as `DLP.py`

## Usage

Run the script:
```bash
python DLP.py
```

When prompted, you'll need to:
1. **Choose activation function**: Enter `relu` or `sigmoid`
2. **Select stopping criterion**:
   - Enter `1` to specify an error threshold
   - Enter `2` to specify a maximum number of epochs

### Example Interaction

```
Choose activation function (relu/sigmoid): relu
What do you want to enter:
 1).error 
2).Epochs: 2
Enter max epochs: 100
```

## Model Architecture

```
Input Layer (13 features after one-hot encoding)
    ↓
Hidden Layer 1 (4 neurons + ReLU/Sigmoid)
    ↓
Hidden Layer 2 (4 neurons + ReLU/Sigmoid)
    ↓
Output Layer (1 neuron - house price prediction)
```

## Hyperparameters

- **Learning Rate**: 0.01
- **Hidden Layer Size**: 4 neurons per layer
- **Batch Size**: 50
- **Train-Test Split**: 70-30
- **Random Seed**: 10 (for reproducibility)

## Training Process

1. **Data Loading**: Reads housing data from CSV
2. **Preprocessing**: 
   - Fills missing values with median
   - One-hot encodes categorical features
   - Normalizes features and targets using z-score
3. **Train-Test Split**: Randomly splits data (70% train, 30% test)
4. **Training Loop**:
   - Mini-batch gradient descent
   - Forward propagation
   - Loss computation (Mean Squared Error)
   - Backpropagation
   - Weight updates
5. **Monitoring**: Displays training and test losses after each epoch

## Output

The model displays:
- Batch-level loss during training
- Epoch-level training and test losses
- Training and testing feature shapes

Example output:
```
Training Features shape: (14448, 13)
Training Targets shape: (14448, 1)
Epoch: 1, Batch: 1, Loss: 0.8234
Epoch: 1, Batch: 2, Loss: 0.7891
...
Epoch: 1, Train Loss: 0.65, Test Loss: 0.68
```

## Project Structure

```
.
├── DLP.py          # Main neural network implementation
├── housing.csv     # California Housing Dataset
└── README.md       # Project documentation
```

## Implementation Details

### Key Components

- **NeuralNetwork Class**: Core implementation with methods for:
  - `activate()`: Applies activation function (ReLU/Sigmoid)
  - `activate_derivative()`: Computes activation function derivatives
  - `forward()`: Forward propagation through the network
  - `backward()`: Backpropagation and gradient computation
  - `compute_loss()`: Mean Squared Error calculation
  - `train()`: Training loop with mini-batch gradient descent

### Weight Initialization

Weights are initialized using uniform distribution in range [-0.1, 0.1] to prevent symmetry and facilitate learning.

### Loss Function

Mean Squared Error (MSE):
```
MSE = (1/n) * Σ(predicted - actual)²
```

## Future Enhancements

- Add early stopping based on validation loss
- Implement additional activation functions (tanh, leaky ReLU)
- Add regularization (L1/L2) to prevent overfitting
- Implement learning rate scheduling
- Add cross-validation
- Visualize training progress and predictions
- Export trained model weights
- Add prediction functionality for new data

## License

This project is created for educational purposes as part of a Deep Learning Programming (DLP) assignment.

## Author

Created as part of DLP Assignment 01

## Acknowledgments

- California Housing Dataset from the StatLib repository
- Built as a learning exercise to understand neural network fundamentals
