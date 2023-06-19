import tensorflow as tf

from neural_models   import create_cnn, create_gan, create_mlp, create_rnn,create_autoencoder, pretrain_model

def compile_model(model):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy')


def create_neural_networks(num_models, supervised=True, num_timesteps = None , data_type = None, input_dim = 0):
    models = []
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)

    
    for _ in range(num_models):
        cnn_model = create_cnn(input_dim, data_type)
        gan_model = create_gan(input_dim, data_type)
        mlp_model = create_mlp(input_dim, data_type)
        rnn_model = create_rnn(input_dim, num_timesteps, data_type)

        if not supervised:
            autoencoder = create_autoencoder(input_dim, data_type)
            cnn_model = pretrain_model(cnn_model, autoencoder)

        combined_model = tf.keras.Sequential([cnn_model, gan_model, mlp_model, rnn_model])

        compile_model(combined_model)

        models.append(combined_model)

    return models



