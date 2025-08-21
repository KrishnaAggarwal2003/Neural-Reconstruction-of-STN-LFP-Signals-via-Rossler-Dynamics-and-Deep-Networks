import tensorflow as tf
from ffrnn import FF

class flip_flop(tf.keras.Model):
    def __init__(self, output_dim, output_size, timesteps, features, num_epochs, learning_rate):
        super(flip_flop, self).__init__()
        self.output_dim = output_dim
        self.output_size = output_size
        self.timesteps= timesteps
        self.features = features
        self.num_epochs = num_epochs
        self.lr = learning_rate

    def forward(self):
        model = tf.keras.Sequential([
            tf.keras.layers.RNN(FF(self.output_dim), input_shape=(self.timesteps, self.features), return_sequences=True),
            tf.keras.layers.Dense(self.output_size, activation='sigmoid')
        ])

        return model



def training_mode(model, data, target, batch_num, learning_rate, num_epochs):
    
    compiler = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=compiler, loss=tf.keras.losses.mean_squared_error)
    
    result = model.fit(data, target, epochs=num_epochs, batch_size = batch_num)
    return result
