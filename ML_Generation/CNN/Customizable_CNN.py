import tensorflow as tf

def create_flexible_cnn_model(
    input_shape,
    conv_layers=[(64, 3)],
    conv_activations='relu',
    pooling_layers=None,  # list of dicts like [{'type': 'max', 'size': 2}, None, ...]
    dropout_mode=None,
    dropout_value=0.0,
    dense_layers=[64],
    dense_activations='relu',
    output_units=1,
    output_activation='sigmoid',
    learning_rate=0.001
):
    """
    Build a customizable 1D CNN model with flexible conv/pooling/dense layers and dropout.

    Parameters:
    - input_shape: Tuple (timesteps, features)
    - conv_layers: List of (filters, kernel_size)
    - conv_activations: str or list of str for each conv layer
    - pooling_layers: List of dicts or None (same length as conv_layers),
                      e.g., [{'type': 'max', 'size': 2}, None, {'type': 'avg', 'size': 2}]
    - dropout_mode: None, 'uniform', 'per_layer', 'every_n'
    - dropout_value: float, list of floats, or (float, n)
    - dense_layers: List of integers for Dense layer sizes
    - dense_activations: str or list of str for dense layers
    - output_units: Output layer size
    - output_activation: Activation for output layer
    - learning_rate: Learning rate for Adam optimizer

    Returns:
    - tf.keras.Model
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))

    # Normalize activations
    if isinstance(conv_activations, str):
        conv_activations = [conv_activations] * len(conv_layers)
    if pooling_layers is None:
        pooling_layers = [None] * len(conv_layers)

    # Dropout per conv layer
    if dropout_mode == 'uniform':
        dropout_rates = [dropout_value] * len(conv_layers)
    elif dropout_mode == 'per_layer':
        dropout_rates = dropout_value
    elif dropout_mode == 'every_n':
        rate, n = dropout_value
        dropout_rates = [rate if (i + 1) % n == 0 else 0.0 for i in range(len(conv_layers))]
    else:
        dropout_rates = [0.0] * len(conv_layers)

    # Add Conv + Pool + Dropout
    for (filters, kernel_size), activation, pool_cfg, drop_rate in zip(conv_layers, conv_activations, pooling_layers, dropout_rates):
        model.add(tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation=activation))
        if pool_cfg:
            if pool_cfg['type'] == 'max':
                model.add(tf.keras.layers.MaxPooling1D(pool_size=pool_cfg['size']))
            elif pool_cfg['type'] == 'avg':
                model.add(tf.keras.layers.AveragePooling1D(pool_size=pool_cfg['size']))
        if drop_rate > 0:
            model.add(tf.keras.layers.Dropout(drop_rate))

    model.add(tf.keras.layers.Flatten())

    # Normalize dense activations
    if isinstance(dense_activations, str):
        dense_activations = [dense_activations] * len(dense_layers)

    for units, act in zip(dense_layers, dense_activations):
        model.add(tf.keras.layers.Dense(units, activation=act))

    model.add(tf.keras.layers.Dense(output_units, activation=output_activation))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = 'categorical_crossentropy' if output_units > 1 else 'binary_crossentropy'
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model

if __name__ == "__main__":
    # Sample shape: 60 time steps, 8 indicators (like Close, EMA, SMA, MACD, etc.)
    input_shape = (60, 8)

    model = create_flexible_cnn_model(
        input_shape=input_shape,
        conv_layers=[(64, 5), (32, 3)],
        conv_activations=['relu', 'tanh'],
        pooling_layers=[{'type': 'max', 'size': 2}, {'type': 'avg', 'size': 2}],
        dropout_mode='per_layer',
        dropout_value=[0.2, 0.3],
        dense_layers=[128, 64],
        dense_activations=['relu', 'relu'],
        output_units=3,
        output_activation='softmax',
        learning_rate=0.0005
    )

    model.summary()