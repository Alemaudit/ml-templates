from tensorflow import keras
import numpy as np


def get_mlp(input_dimension: int, hidden_nodes: int, output_dimension: int) -> keras.models.Model:
    inputs = keras.layers.Input(
        shape=(input_dimension, ),
        name='input'
    )
    hidden_layer = keras.layers.Dense(
        units=hidden_nodes,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(0.0001),
        name='hidden_layer'
    )(inputs)
    output = keras.layers.Dense(
        units=output_dimension,
        activation='linear',
        name='output'
    )(hidden_layer)
    return keras.models.Model(inputs=inputs, outputs=output, name='mlp')


if __name__ == "__main__":
    input_dimension = 10
    output_dimension = 2
    training_samples = 10000
    batch_size = 1000

    X = np.random.random((training_samples, input_dimension))
    A = np.random.random((input_dimension, output_dimension))
    Y = X @ A

    model = get_mlp(input_dimension=input_dimension, hidden_nodes=30, output_dimension=output_dimension)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='mean_squared_error',
        min_delta=0.001,
        patience=5,
        verbose=1,
        mode='auto',
        baseline=None,
        restore_best_weights=True
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.MeanSquaredError(),
        metrics=['mean_squared_error'],
    )
    training_history = model.fit(
        X, Y, epochs=200, batch_size=batch_size, verbose=1, callbacks=[early_stopping]
    )
    print(model.predict(X))
    print(Y)
