import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

IMG_HEIGHT = 256
IMG_WIDTH = 256
LAMBDA = 10


#------------------ Generator-------------------------
def downsample(filters, size, apply_instancenorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_instancenorm:
        result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    result.add(layers.LeakyReLU())

    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

    result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    if apply_dropout:
        result.add(layers.Dropout(0.5))

    result.add(layers.ReLU())

    return result

#-------------------Loss Functions-----------------------------
def discriminator_loss_fn(real, generated):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(real), real)

    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5

def generator_loss_fn(generated):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(generated), generated)

def calc_cycle_loss_fn(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
  return LAMBDA * loss1

def identity_loss_fn(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss

#-------------------CycleGAN Keras Model-----------------------------

class CycleGAN(tf.keras.Model):
    def __init__(self, lambda_cycle =10):
        super(CycleGAN, self).__init__()
        self.generator_G = self.build_generator(id='Pet2Art') # Pets to art-styled pet images
        self.generator_F = self.build_generator(id ='Art2Pet') # Reverse generator
        self.discriminator_X = self.build_discriminator(id='pet') # Pet domain discriminator
        self.discriminator_Y = self.build_discriminator(id='art') # Art domain discriminator
        self.lambda_cycle = lambda_cycle
        self.optimizers = self.get_optimizers()

    def call(self, x):
        return self.generator_G(x)

    def compile(self):
        super(CycleGAN, self).compile()

        # Compilation config
        config = self.optimizers

        self.gen_G_optimizer = config['generator_G_optimizer']
        self.gen_F_optimizer = config['generator_F_optimizer']
        self.disc_X_optimizer = config['discriminator_X_optimizer']
        self.disc_Y_optimizer = config['discriminator_Y_optimizer']
        self.gen_loss_fn = generator_loss_fn
        self.disc_loss_fn = discriminator_loss_fn
        self.cycle_loss_fn = calc_cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn

    def train_step(self, real_x, real_y):

        with tf.GradientTape(persistent=True) as tape:
            # Generate fake images
            fake_y = self.generator_G(real_x)
            fake_x = self.generator_F(real_y)

            # Generate reconstructed images
            reconstructed_x = self.generator_F(fake_y)
            reconstructed_y = self.generator_G(fake_x)

            # Generate identity images
            same_x = self.generator_F(real_x)
            same_y = self.generator_G(real_y)

            # Discriminator outputs
            disc_real_x = self.discriminator_X(real_x)
            disc_fake_x = self.discriminator_X(fake_x)
            disc_real_y = self.discriminator_Y(real_y)
            disc_fake_y = self.discriminator_Y(fake_y)

            # Define and calculate losses
            gen_G_loss = self.gen_loss_fn(disc_fake_y)
            gen_F_loss = self.gen_loss_fn(disc_fake_x)
            cycle_loss = self.cycle_loss_fn(real_x, reconstructed_x) + self.cycle_loss_fn(real_y, reconstructed_y)
            identity_loss_g = self.identity_loss_fn(real_y, same_y)
            identity_loss_f = self.identity_loss_fn(real_x, same_x)


            # Aggregate losses
            total_gen_G_loss = gen_G_loss + cycle_loss + identity_loss_g
            total_gen_F_loss = gen_F_loss + cycle_loss + identity_loss_f
            disc_X_loss = self.disc_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = self.disc_loss_fn(disc_real_y, disc_fake_y)

        # Compute gradients and update weights
        gen_G_gradients = tape.gradient(total_gen_G_loss, self.generator_G.trainable_variables)
        self.gen_G_optimizer.apply_gradients(zip(gen_G_gradients, self.generator_G.trainable_variables))

        gen_F_gradients = tape.gradient(total_gen_F_loss, self.generator_F.trainable_variables)
        self.gen_F_optimizer.apply_gradients(zip(gen_F_gradients, self.generator_F.trainable_variables))

        disc_X_gradients = tape.gradient(disc_X_loss, self.discriminator_X.trainable_variables)
        self.disc_X_optimizer.apply_gradients(zip(disc_X_gradients, self.discriminator_X.trainable_variables))

        disc_Y_gradients = tape.gradient(disc_Y_loss, self.discriminator_Y.trainable_variables)
        self.disc_Y_optimizer.apply_gradients(zip(disc_Y_gradients, self.discriminator_Y.trainable_variables))

        return {
            "gen_G_loss": total_gen_G_loss,
            "gen_F_loss": total_gen_F_loss,
            "disc_X_loss": disc_X_loss,
            "disc_Y_loss": disc_Y_loss,
        }

    # Generator model setup
    def build_generator(self, id):
        inputs = layers.Input(shape=[IMG_WIDTH,IMG_HEIGHT,3])

        down_stack = [
            downsample(64, 4, apply_instancenorm=False), 
            downsample(128, 4), 
            downsample(256, 4), 
            downsample(512, 4), 
            downsample(512, 4), 
            downsample(512, 4), 
            downsample(512, 4), 
            downsample(512, 4), 
        ]

        up_stack = [
            upsample(512, 4, apply_dropout=True),
            upsample(512, 4, apply_dropout=True), 
            upsample(512, 4, apply_dropout=True), 
            upsample(512, 4), 
            upsample(256, 4), 
            upsample(128, 4), 
            upsample(64, 4), 
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = layers.Conv2DTranspose(3, 4,
                                  strides=2,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  activation='tanh') 
        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = layers.Concatenate()([x, skip])

        x = last(x)

        return keras.Model(inputs=inputs, outputs=x, name = f"Generator_{id}")

    # Discriminator model setup
    def build_discriminator(self, id):
        initializer = tf.random_normal_initializer(0., 0.02)
        gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

        inp = layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, 3], name='input_image')

        x = inp

        down1 = downsample(64, 4, False)(x) 
        down2 = downsample(128, 4)(down1) 
        down3 = downsample(256, 4)(down2) 

        zero_pad1 = layers.ZeroPadding2D()(down3) 
        conv = layers.Conv2D(512, 4, strides=1,
                         kernel_initializer=initializer,
                         use_bias=False)(zero_pad1) 

        norm1 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(conv)

        leaky_relu = layers.LeakyReLU()(norm1)

        zero_pad2 = layers.ZeroPadding2D()(leaky_relu) 

        last = layers.Conv2D(1, 4, strides=1,
                         kernel_initializer=initializer)(zero_pad2) 

        return tf.keras.Model(inputs=inp, outputs=last, name = f"Discriminator_{id}")
    
    # Compilation optimizers defaults
    def get_optimizers(self):
        config = {
          'generator_G_optimizer': tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
          'generator_F_optimizer': tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
          'discriminator_X_optimizer': tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
          'discriminator_Y_optimizer': tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        }

        return config
    
    # Set model optimizers manually before compilation
    def set_optimizers(self, 
                       generator_G_optimizer,
                       generator_F_optimizer,
                       discriminator_X_optimizer,
                       discriminator_Y_optimizer):
        config = {
          'generator_G_optimizer': generator_G_optimizer,
          'generator_F_optimizer': generator_F_optimizer,
          'discriminator_X_optimizer': discriminator_X_optimizer,
          'discriminator_Y_optimizer': discriminator_Y_optimizer
        }

        self.optimizers = config

if __name__ == "__main__":
    # Instantiate and compile the CycleGAN model
    cycle_gan = CycleGAN()
    cycle_gan.compile()

    # Test model forward-pass
    import numpy as np
    cycle_gan(np.random.rand(1,256,256,3))
    
    # Print summary
    cycle_gan.summary()


