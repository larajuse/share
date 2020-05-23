import tensorflow as tf

class GAN():
    def __init__(self, generator, discriminator, optimizer_gen, optimizer_dis,
                 batch_size, latent_dim):
        self.__generator = generator
        self.__discriminator = discriminator
        self.__optimizer_gen = optimizer_gen
        self.__optimizer_dis = optimizer_dis
        self.__batch_size = batch_size
        self.__latent_dim = latent_dim
        
        self.__gen_loss_metric = tf.keras.metrics.Mean(name='gen_loss_mean')
        self.__disc_loss_metric = tf.keras.metrics.Mean(name='disc_loss_mean')
        self.__real_acc_metric =  tf.keras.metrics.Mean(name='real_acc_mean')
        self.__fake_acc_metric =  tf.keras.metrics.Mean(name='fake_acc_mean')
    def __discriminator_loss(self, real, fake):
        loss_real = tf.losses.binary_crossentropy(0.9*tf.ones_like(real), real)
        loss_fake = tf.losses.binary_crossentropy(tf.zeros_like(fake), fake)
        return loss_real+loss_fake
    def __generator_loss(self, fake):
        return tf.losses.binary_crossentropy(tf.ones_like(fake), fake)
    @tf.function
    def __step(self, batch):
        
        latent_vecs = tf.random.normal([self.__batch_size, self.__latent_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = self.__generator(latent_vecs, training=True)
            fake_preds = self.__discriminator(fake_images, training=True)
            real_preds = self.__discriminator(batch, training=True)            
            loss_generator = self.__generator_loss(fake_preds)
            loss_discriminator = self.__discriminator_loss(real_preds, fake_preds)
            
            self.__gen_loss_metric(loss_generator)
            self.__disc_loss_metric(loss_discriminator)
            self.__real_acc_metric(tf.cast(tf.equal(tf.ones_like(real_preds), tf.math.round(real_preds)), tf.float32))
            self.__fake_acc_metric(tf.cast(tf.equal(tf.zeros_like(fake_preds), tf.math.round(fake_preds)), tf.float32))
            
        generator_grads = gen_tape.gradient(loss_generator, self.__generator.trainable_variables)
        discriminator_grads = disc_tape.gradient(loss_discriminator, self.__discriminator.trainable_variables)

        self.__optimizer_gen.apply_gradients(zip(generator_grads, self.__generator.trainable_variables))
        self.__optimizer_dis.apply_gradients(zip(discriminator_grads, self.__discriminator.trainable_variables))
        return 
        
    def fit(self, gen, epochs, steps_per_epoch):
        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                batch = next(gen)[0]
                self.__step(batch)
            print(f"Epoch {epoch+1}/{epochs} Generator loss: {self.__gen_loss_metric.result():.4f} Discriminator loss {self.__disc_loss_metric.result():.4f}")
            print(f"Accuracy on real examples: {self.__real_acc_metric.result():.4f}, Accuracy on fake examples: {self.__fake_acc_metric.result():.4f}")