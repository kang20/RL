import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(128, activation='relu')
        self.fc3 = tf.keras.layers.Dense(128, activation='relu')
        self.fc4 = tf.keras.layers.Dense(1, use_bias=False)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


def true_fun(X):
    noise = np.random.rand(X.shape[0]) * 0.4 - 0.2
    return np.cos(1.5 * np.pi * X) + X + noise


def plot_results(model):
    x = np.linspace(0, 5, 100)
    input_x = tf.convert_to_tensor(x, dtype=tf.float32)
    input_x = tf.expand_dims(input_x, axis=1)
    plt.plot(x, true_fun(x), label="Truth")
    plt.plot(x, model(input_x), label="Prediction")
    plt.legend(loc='lower right', fontsize=15)
    plt.xlim((0, 5))
    plt.ylim((-1, 5))
    plt.grid()


def main():
    data_x = np.random.rand(10000) * 5
    validation_x = np.linspace(0, 5, 100)
    validation_y = true_fun(validation_x)

    model = Model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    mse_loss = tf.keras.losses.MeanSquaredError()

    train_losses = []
    val_losses = []

    for step in range(10000):
        batch_x = np.random.choice(data_x, 32)
        batch_x_tensor = tf.convert_to_tensor(batch_x, dtype=tf.float32)
        batch_x_tensor = tf.expand_dims(batch_x_tensor, axis=1)
        batch_y = true_fun(batch_x)
        truth = tf.convert_to_tensor(batch_y, dtype=tf.float32)
        truth = tf.expand_dims(truth, axis=1)

        with tf.GradientTape() as tape:
            pred = model(batch_x_tensor)
            loss = mse_loss(truth, pred)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_losses.append(loss.numpy())

        if step % 100 == 0:
            val_x_tensor = tf.convert_to_tensor(validation_x, dtype=tf.float32)
            val_x_tensor = tf.expand_dims(val_x_tensor, axis=1)
            val_pred = model(val_x_tensor)
            val_loss = mse_loss(tf.convert_to_tensor(validation_y, dtype=tf.float32), val_pred)
            val_losses.append(val_loss.numpy())
            print(f"Step {step}, Training Loss: {loss.numpy()}, Validation Loss: {val_loss.numpy()}")

    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(np.arange(0, 10000, 100), val_losses, label='Validation Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    plot_results(model)
    plt.show()


if __name__ == "__main__":
    main()