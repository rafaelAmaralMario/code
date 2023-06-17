import numpy as np
import tensorflow as tf
from plasticity_rules import HebbianPlasticity, HebbianModifiedPlasticity, STDPPlasticity, eSTDPPlasticity, OjaPlasticity, BCMPlasticity

def create_pln_layer(input_dim, output_dim, input_length):
    model = tf.keras.Sequential()
    model.add(Embedding(input_dim, output_dim, input_length=input_length))
    # Adicione outras camadas de PLN aqui, se necessário
    return model

# Função para criar um modelo CNN
def create_cnn(input_dim, data_type):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(input_dim, input_dim, 1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # Adicionar mais 15 camadas com tamanhos aleatórios
    random_sizes = np.random.randint(low=2, high=32768, size=30)
    for size in random_sizes:
        model.add(tf.keras.layers.Conv2D(size, kernel_size=(3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Aplicar a plasticidade sináptica na camada desejada
    if data_type == "Arquivos" or data_type == "Texto":
        model.add(HebbianPlasticity(units=128, activation='relu', learning_rate=0.001))
        model.add(HebbianModifiedPlasticity(units=128, activation='relu', learning_rate=0.001))
    elif data_type == "Áudio" or data_type == "Fala":
        model.add(STDPPlasticity(units=128, activation='relu', learning_rate=0.001, pre_tau=20, post_tau=20))
        model.add(eSTDPPlasticity(units=128, activation='relu', learning_rate=0.001, pre_tau_pos=20, pre_tau_neg=20,
                                  post_tau_pos=20, post_tau_neg=20))
    elif data_type == "Imagens" or data_type == "Vídeos":
        model.add(HebbianPlasticity(units=128, activation='relu', learning_rate=0.001))
        model.add(OjaPlasticity(units=128, activation='relu', learning_rate=0.001))
        model.add(BCMPlasticity(units=128, activation='relu', learning_rate=0.001))

    model.add(tf.keras.layers.Reshape((-1, model.output_shape[3])))
    model.add(tf.keras.layers.LSTM(128, return_sequences=True))
    model.add(tf.keras.layers.Flatten())
    return model

# Função para criar um modelo GAN
def create_gan(input_dim, data_type):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(512, activation='relu', input_dim=input_dim))
    
    # Adicionar mais 15 camadas com tamanhos aleatórios
    random_sizes = np.random.randint(low=2, high=1024, size=30)
    for size in random_sizes:
        model.add(tf.keras.layers.Dense(size, activation='relu'))

    if data_type == "Arquivos" or data_type == "Texto":
        model.add(HebbianPlasticity(units=64, activation='relu', learning_rate=0.001))
        model.add(HebbianModifiedPlasticity(units=64, activation='relu', learning_rate=0.001))
    elif data_type == "Áudio" or data_type == "Fala":
        model.add(STDPPlasticity(units=64, activation='relu', learning_rate=0.001, pre_tau=20, post_tau=20))
        model.add(eSTDPPlasticity(units=64, activation='relu', learning_rate=0.001, pre_tau_pos=20, pre_tau_neg=20,
                                  post_tau_pos=20, post_tau_neg=20))
    elif data_type == "Imagens":
        model.add(HebbianPlasticity(units=64, activation='relu', learning_rate=0.001))
        model.add(OjaPlasticity(units=64, activation='relu', learning_rate=0.001))
        model.add(BCMPlasticity(units=64, activation='relu', learning_rate=0.001))
    elif data_type == "Vídeos":
        model.add(HebbianPlasticity(units=64, activation='relu', learning_rate=0.001))
        model.add(OjaPlasticity(units=64, activation='relu', learning_rate=0.001))
        model.add(BCMPlasticity(units=64, activation='relu', learning_rate=0.001))

    model.add(tf.keras.layers.Dense(input_dim, activation='tanh'))
    return model

# Função para criar um modelo MLP com regras de plasticidade baseadas no tipo de arquivo
def create_mlp(input_dim, data_type):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu', input_dim=input_dim))
    
            # Adicionar mais 15 camadas com tamanhos aleatórios
    random_sizes = np.random.randint(low=2, high=512, size=30)
    for size in random_sizes:
        model.add(tf.keras.layers.Dense(size, activation='relu'))


    if data_type == "Arquivos" or data_type == "Texto":
        model.add(HebbianPlasticity(units=128, activation='relu', learning_rate=0.001))
        model.add(HebbianModifiedPlasticity(units=128, activation='relu', learning_rate=0.001))
    elif data_type == "Áudio" or data_type == "Fala":
        model.add(STDPPlasticity(units=128, activation='relu', learning_rate=0.001, pre_tau=20, post_tau=20))
        model.add(eSTDPPlasticity(units=128, activation='relu', learning_rate=0.001, pre_tau_pos=20, pre_tau_neg=20,
                                  post_tau_pos=20, post_tau_neg=20))
    elif data_type == "Imagens" or data_type == "Vídeos":
        model.add(HebbianPlasticity(units=128, activation='relu', learning_rate=0.001))
        model.add(OjaPlasticity(units=128, activation='relu', learning_rate=0.001))
        model.add(BCMPlasticity(units=128, activation='relu', learning_rate=0.001))
    
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

# Função para criar um modelo RNN com regras de plasticidade baseadas no tipo de arquivo
def create_rnn(input_dim, num_timesteps, data_type):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.SimpleRNN(64, input_shape=(num_timesteps, input_dim)))

        # Adicionar mais 15 camadas com tamanhos aleatórios
    random_sizes = np.random.randint(low=2, high=512, size=30)
    for size in random_sizes:
        model.add(tf.keras.layers.SimpleRNN(size, return_sequences=True))

    model.add(tf.keras.layers.SimpleRNN(8))
    if data_type == "Arquivos" or data_type == "Texto":
        model.add(HebbianPlasticity(units=128, activation='relu', learning_rate=0.001))
        model.add(HebbianModifiedPlasticity(units=128, activation='relu', learning_rate=0.001))
    elif data_type == "Áudio" or data_type == "Fala":
        model.add(STDPPlasticity(units=128, activation='relu', learning_rate=0.001, pre_tau=20, post_tau=20))
        model.add(eSTDPPlasticity(units=128, activation='relu', learning_rate=0.001, pre_tau_pos=20, pre_tau_neg=20,
                                  post_tau_pos=20, post_tau_neg=20))
    elif data_type == "Imagens" or data_type == "Vídeos":
        model.add(HebbianPlasticity(units=128, activation='relu', learning_rate=0.001))
        model.add(OjaPlasticity(units=128, activation='relu', learning_rate=0.001))
        model.add(BCMPlasticity(units=128, activation='relu', learning_rate=0.001))
    
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

def create_autoencoder(input_dim, data_type):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(512, activation='relu', input_dim=input_dim))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))

    # Adicionar mais 15 camadas com tamanhos aleatórios
    random_sizes = np.random.randint(low=2, high=512, size=30)
    for size in random_sizes:
        model.add(tf.keras.layers.Dense(size, activation='relu'))

    if data_type == "Arquivos" or data_type == "Texto":
        model.add(HebbianPlasticity(units=64, activation='relu', learning_rate=0.001))
        model.add(HebbianModifiedPlasticity(units=64, activation='relu', learning_rate=0.001))
    elif data_type == "Áudio" or data_type == "Fala":
        model.add(STDPPlasticity(units=64, activation='relu', learning_rate=0.001, pre_tau=20, post_tau=20))
        model.add(eSTDPPlasticity(units=64, activation='relu', learning_rate=0.001, pre_tau_pos=20, pre_tau_neg=20,
                                  post_tau_pos=20, post_tau_neg=20))
    elif data_type == "Imagens":
        model.add(HebbianPlasticity(units=64, activation='relu', learning_rate=0.001))
        model.add(OjaPlasticity(units=64, activation='relu', learning_rate=0.001))
        model.add(BCMPlasticity(units=64, activation='relu', learning_rate=0.001))
    elif data_type == "Vídeos":
        model.add(HebbianPlasticity(units=64, activation='relu', learning_rate=0.001))
        model.add(OjaPlasticity(units=64, activation='relu', learning_rate=0.001))
        model.add(BCMPlasticity(units=64, activation='relu', learning_rate=0.001))

    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(input_dim, activation='tanh'))
    
    return model

def pretrain_model(cnn_model, autoencoder):
    autoencoder.trainable = False

    for i in range(1, min(len(autoencoder.layers), 9)):
        cnn_model.layers[-i].set_weights(autoencoder.layers[4 + i].get_weights())
    
    return cnn_model
