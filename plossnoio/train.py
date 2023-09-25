import tensorflow as tf
from keras import backend as K
import openl3
import numpy as np
import os
from keras.preprocessing.image import img_to_array

input_repr, content_type, embedding_size = 'linear', 'music', 6144
input_repr2, content_type2, embedding_size2 = 'mel256', 'music', 512
model = openl3.models.load_audio_embedding_model(input_repr2, content_type2, embedding_size2, frontend='librosa')

def reshape_and_pad_tensor(tensor, target_shape=(257, 197)):
    batch_size, height, width, channels = tensor.shape
    zero_padding = np.zeros((batch_size, 1, width, channels))
    tensor = np.concatenate([tensor, zero_padding], axis=1)

    reshaped_tensors = []

    for i in range(batch_size):
        sample = tensor[i]
        segments = []

        for j in range(0, sample.shape[1], target_shape[1]):
            segment = sample[:, j:j + target_shape[1], :]
            padding = np.zeros((target_shape[0], target_shape[1] - segment.shape[1], segment.shape[2]))
            segment = np.concatenate([segment, padding], axis=1)
            segments.append(segment)

        reshaped_tensor = np.stack(segments)
        reshaped_tensors.append(reshaped_tensor)

    output_tensor = np.stack(reshaped_tensors)

    return output_tensor

def get_audio_embeddings(real_x, fake_image):
    reshaped_real_x = reshape_and_pad_tensor(real_x)
    reshaped_fake_image = reshape_and_pad_tensor(fake_image)

    reshaped_real_x_flat = reshaped_real_x.reshape(-1, 257, 197, 1)
    reshaped_fake_image_flat = reshaped_fake_image.reshape(-1, 257, 197, 1)

    real_emb = model(reshaped_real_x_flat, training=False)
    fake_emb = model(reshaped_fake_image_flat, training=False)

    return real_emb, fake_emb


# Function to reshape and pad the tensor to the target shape (256, 199, 1)
def reshape_and_pad_tensor2(tensor, target_shape=(256, 199)):
    batch_size, height, width, channels = tensor.shape
    reshaped_tensors = []

    for i in range(batch_size):
        sample = tensor[i]
        segments = []

        # Segment the tensor based on target width
        for j in range(0, width, target_shape[1]):
            segment = sample[:, j:j + target_shape[1], :]

            # Zero-padding
            padding_width = target_shape[1] - segment.shape[1]
            padding = np.zeros((target_shape[0], padding_width, channels))
            segment = np.concatenate([segment, padding], axis=1)

            segments.append(segment)

        reshaped_tensor = np.stack(segments)
        reshaped_tensors.append(reshaped_tensor)

    output_tensor = np.concatenate(reshaped_tensors, axis=0)
    return output_tensor


def get_audio_embeddings2(real_x, fake_image):
    reshaped_real_x = reshape_and_pad_tensor2(real_x)
    reshaped_fake_image = reshape_and_pad_tensor2(fake_image)

    # Flatten the tensors, if needed you can further reshape them
    reshaped_real_x_flat = reshaped_real_x
    reshaped_fake_image_flat = reshaped_fake_image

    real_emb = model(reshaped_real_x_flat, training=False)
    fake_emb = model(reshaped_fake_image_flat, training=False)

    return real_emb, fake_emb




@tf.function
def WGAN_GP_train_d_step(real_x,
                         noisy_x,
                         discriminator,
                         generator,
                         discriminator_optimizer,
                         LAMBDA,
                         batch_size):

    epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=1)
    ###################################
    # Train D
    ###################################
    with tf.GradientTape(persistent=True) as d_tape:
        with tf.GradientTape() as gp_tape:
            fake_image = generator(noisy_x, training=True)
            fake_image_mixed = epsilon * tf.dtypes.cast(real_x, tf.float32) + ((1 - epsilon) * fake_image)
            fake_mixed_pred = discriminator(fake_image_mixed, training=True)

        # Compute gradient penalty
        grads = gp_tape.gradient(fake_mixed_pred, fake_image_mixed)
        grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1))

        fake_pred = discriminator(fake_image, training=True)
        real_pred = discriminator(real_x, training=True)

        D_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred) + LAMBDA * gradient_penalty
    # Calculate the gradients for discriminator
    D_gradients = d_tape.gradient(D_loss,
                                  discriminator.trainable_variables)
    # Apply the gradients to the optimizer
    discriminator_optimizer.apply_gradients(zip(D_gradients,
                                    discriminator.trainable_variables))

    return D_loss, fake_image

@tf.function
def WGAN_GP_train_g_step(real_x, noisy_x, real_embeddings, fake_embeddings, discriminator, generator, generator_optimizer, gamma=0.5, zeta=0.25):
    ###################################
    # Train G
    ###################################
    with tf.GradientTape() as g_tape:
        #Crtitic score for fake image
        fake_image = generator(noisy_x, training=True)
        fake_pred = discriminator(fake_image, training=True)

        # Compute perceptual loss
        perceptual_loss = tf.reduce_mean(tf.square(real_embeddings - fake_embeddings))
        # Compute reconstruction loss
        recon_loss = tf.reduce_mean(tf.square(real_x - fake_image))  # MSE
        # Compute WGAN loss
        g_loss = -tf.reduce_mean(fake_pred)

        # Combined loss
        total_g_loss = g_loss + gamma * recon_loss + zeta * perceptual_loss

    # Calculate the gradients for generator
    g_gradients = g_tape.gradient(total_g_loss,
                                  generator.trainable_variables)
    # Apply the gradients to the optimizer
    generator_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

    return total_g_loss, recon_loss, g_loss, perceptual_loss, fake_image


def learning_rate_decay(current_lr, decay_factor, MIN_LR):
    '''
        Calculate new learning rate using decay factor
    '''
    new_lr = max(current_lr / decay_factor, MIN_LR)
    return new_lr

def set_learning_rate(new_lr, optimizer):
    '''
        Set new learning rate to optimizers
    '''
    K.set_value(optimizer.lr, new_lr)