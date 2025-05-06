import tensorflow as tf
import numpy as np

def create_model(input_shape,
                 hidden_layer_sizes,
                 hidden_layer_activations='relu',
                 output_units=1,
                 output_activation='sigmoid',
                 dropout_mode=None,
                 dropout_value=0.0,
                 learning_rate=0.001):  # Add learning_rate parameter
    """
    Create a customizable fully connected neural network model with flexible architecture,
    activation functions, and dropout configurations.

    Parameters:
    - input_shape (int):
        Number of features in the input data. This defines the shape of the input layer.

    - hidden_layer_sizes (list of int):
        A list where each element represents the number of units in a hidden layer.
        The length of this list determines the number of hidden layers in the network.

    - hidden_layer_activations (str or list of str, default='relu'):
        Activation function(s) for the hidden layers. Can be a single string (applied to all layers)
        or a list of strings (each corresponding to a specific hidden layer).

    - output_units (int, default=1):
        Number of units in the output layer. Common values are:
            * 1 for binary classification or regression
            * >1 for multi-class classification (e.g., softmax output)

    - output_activation (str, default='sigmoid'):
        Activation function for the output layer. Use 'sigmoid' for binary classification,
        'softmax' for multi-class classification, or 'linear' for regression.

    - dropout_mode (str or None, default=None):
        Controls how dropout is applied to the network:
            * None: No dropout
            * 'uniform': Apply the same dropout rate to all hidden layers
            * 'per_layer': Specify a different dropout rate for each hidden layer
            * 'every_n': Apply dropout to every n-th layer only

    - dropout_value (float, list of float, or tuple, depending on mode):
        Specifies the dropout rate(s), depending on the chosen dropout_mode:
            * If 'uniform': A float (e.g., 0.2) to apply to all layers
            * If 'per_layer': A list of floats, one for each hidden layer
            * If 'every_n': A tuple (rate: float, n: int), where dropout is applied every n-th layer

    - learning_rate (float, default=0.001):
        Learning rate for the optimizer.

    Returns:
    - tf.keras.Model: A compiled Sequential model with the specified configuration.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_shape,)))

    # Ensure activation functions list matches layer sizes
    if isinstance(hidden_layer_activations, str):
        hidden_layer_activations = [hidden_layer_activations] * len(hidden_layer_sizes)

    # If dropout_mode is 'uniform', create a list with same rate
    if dropout_mode == 'uniform':
        dropout_rates = [dropout_value] * len(hidden_layer_sizes)
    elif dropout_mode == 'per_layer':
        dropout_rates = dropout_value  # Expecting list
    elif dropout_mode == 'every_n':
        dropout_rates = []
        rate, n = dropout_value
        for i in range(len(hidden_layer_sizes)):
            dropout_rates.append(rate if (i + 1) % n == 0 else 0.0)
    else:
        dropout_rates = [0.0] * len(hidden_layer_sizes)

    # Build layers
    for size, activation, drop in zip(hidden_layer_sizes, hidden_layer_activations, dropout_rates):
        model.add(tf.keras.layers.Dense(size, activation=activation))
        if drop > 0:
            model.add(tf.keras.layers.Dropout(drop))

    model.add(tf.keras.layers.Dense(output_units, activation=output_activation))

    # Compile model with Adam optimizer and the specified learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model


# Optional: test run
if __name__ == "__main__":
    # Example usage 1: Custom layer sizes and activations
    hidden_layer_sizes_1 = [25, 15, 10]
    hidden_layer_activations = ['relu', 'tanh', 'relu']
    model1 = create_model(5,
                          hidden_layer_sizes_1,
                          hidden_layer_activations,
                          output_units=1,
                          output_activation='sigmoid',
                          dropout_mode='per_layer',
                          dropout_value=[0.1, 0.3, 0.0],
                          learning_rate=0.0005
                          )
    model1.summary()

    # Example usage 2: Auto-generated layer sizes with sigmoid output
    hidden_layer_sizes_2 = np.arange(10, 110, 10)
    model2 = create_model(5,
                          hidden_layer_sizes_2,
                          hidden_layer_activations='relu',
                          output_units=1,
                          output_activation='sigmoid',
                          dropout_mode='uniform',
                          dropout_value=0.2,
                          learning_rate=0.05
                          )
    model2.summary()