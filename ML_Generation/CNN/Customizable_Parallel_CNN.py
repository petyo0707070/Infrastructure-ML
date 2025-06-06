from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, AveragePooling1D, Dropout, Flatten, Dense, Concatenate

def build_parallel_cnn_model(
    input_shape,
    conv_branches=[{"filters": 64, "kernel_size": 5}],
    pool_type="max",  # "max" or "avg"
    pool_size=2,
    dropout_rate=0.3,
    dense_layer_dropout_rate = 0.3,
    dense_layers=[{"units": 128, "activation": "relu", 'kernel_regularizer': None}],
    output_units=3,
    output_activation="softmax",
    padding="same"  # User chooses whether to use padding
):
    # Check if padding is needed based on kernel size
    for branch_layers in conv_branches:
        for layer in branch_layers:
            kernel_size = layer["kernel_size"]
            if padding == "valid" and (input_shape[0] - kernel_size + 1) < 1:
                raise ValueError(
                    f"Padding 'valid' cannot be used with kernel_size={kernel_size} on input_shape={input_shape}. "
                    "Choose 'same' padding instead."
                )

    input_layer = Input(shape=input_shape)
    branches = []

    for branch_layers in conv_branches:
        x = input_layer
        for layer in branch_layers:
            # Validate kernel size and padding
            kernel_size = layer["kernel_size"]
            #if padding == "valid" and (x.shape[1] is not None) and ((x.shape[1] - kernel_size + 1) < 1):
            #    raise ValueError(f"Invalid padding with kernel_size={kernel_size} on input_shape={x.shape}")

            x = Conv1D(
                filters=layer["filters"],
                kernel_size=kernel_size,
                activation="relu",
                padding=padding
            )(x)

            if pool_type == "max":
                x = MaxPooling1D(pool_size=pool_size)(x)
            elif pool_type == "avg":
                x = AveragePooling1D(pool_size=pool_size)(x)

            if dropout_rate > 0:
                x = Dropout(dropout_rate)(x)

        branches.append(x)

    x = Concatenate()(branches) if len(branches) > 1 else branches[0]
    x = Flatten()(x)

    for dense_layer in dense_layers:

        if dense_layer.get('kernel_regulazer', None) != None: # This runs if we have specified a kernel_regularizer parameter while building the model
            from tensorflow.keras import regularizers
            x = Dense(dense_layer["units"], activation=dense_layer["activation"], kernel_regularizer = dense_layers['kernel_regularizer'])(x)
        else:
            x = Dense(dense_layer["units"], activation=dense_layer["activation"])(x)
            # Add Dropout after each Dense layer (if dense_laer_dropout_rate > 0)
        if dense_layer_dropout_rate > 0:
            x = Dropout(dense_layer_dropout_rate)(x)  # Add dropout here for regularization

    output_layer = Dense(output_units, activation=output_activation)(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


if __name__ == "__main__":
    # Sample shape: 60 time steps, 6 indicators (like Close, EMA, SMA, MACD, etc.)
    model = build_parallel_cnn_model(
        input_shape=(60, 6),
        conv_branches=[
            {"filters": 64, "kernel_size": 3},
            {"filters": 32, "kernel_size": 5},
            {"filters": 16, "kernel_size": 7}
        ],
        pool_type="max",
        pool_size=2,
        dropout_rate=0.3,
        dense_layers=[
            {"units": 128, "activation": "relu"},
            {"units": 64, "activation": "relu"}
        ],
        output_units=3,
        output_activation="softmax",
        padding="same"  # User can specify 'same' or 'valid'
    )
    model.summary()