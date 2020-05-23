import tensorflow as tf
class VAE():
    def __init__(self, encoder, decoder, latent_dim, input_shape, loss_params=[1, 1]):
        self.__encoder = encoder
        self.__decoder = decoder
        self.__latent_dim = latent_dim
        self.__input_shape = input_shape
        self.__loss_params = loss_params
    @tf.function
    def sample(self, params):
        z_mean, z_log_var = params
        # Reparametrization trick
        epsilon = tf.random.normal(shape=(32, z_mean.shape[1]))
        return z_mean+tf.exp(0.5*z_log_var)*epsilon
    def compile_model(self):
        # Defining complete model
        inp = tf.keras.layers.Input(shape=self.__input_shape)
        z_mean, z_log_var = self.__encoder(inp)
        z = tf.keras.layers.Lambda(self.sample)([z_mean, z_log_var])
        reconstruction = self.__decoder(z)
        self.model = tf.keras.Model(inputs=[inp], outputs=[reconstruction])
        # Defining a reconstruction loss
        loss1 = tf.reduce_sum(tf.losses.mse(inp, reconstruction), axis=[1, 2])*self.__loss_params[0]
        # Defining the vae loss
        loss2 = 1+z_log_var-(z_mean)**2-tf.exp(z_log_var)
        loss2 = -tf.reduce_sum(loss2, axis=-1)*self.__loss_params[1]
        
        loss = loss1+loss2
        self.model.add_loss(loss)
        self.model.compile(optimizer="adam")
        