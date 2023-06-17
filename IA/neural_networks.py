import tensorflow as tf
import magic
from neural_models   import create_cnn, create_gan, create_mlp, create_rnn,create_autoencoder, pretrain_model
from utils   import calculate_num_timesteps_audio, calculate_num_timesteps_video, calculate_num_timesteps_image, calculate_num_timesteps_text,calculate_num_timesteps_file

def compile_model(model):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy')

def get_data_type(input_data):
    file_type = magic.from_buffer(input_data, mime=True)

    if file_type.startswith('image'):
        return 'Imagens', 'image'
    elif file_type.startswith('video'):
        return 'Vídeos', 'video'
    elif file_type.startswith('audio'):
        return 'Áudio', 'audio'
    elif file_type.startswith('text'):
        return 'Texto', 'text'
    else:
        return 'Arquivos', 'file'

def create_neural_networks(input_data, num_models, supervised=True):
    models = []
    data_type, data_subtype = get_data_type(input_data)
    input_dim = input_data.shape[1]

    num_timesteps = None

    if data_subtype == 'audio':
        # Calcular o num_timesteps específico para dados de áudio
        num_timesteps = calculate_num_timesteps_audio(input_data)
    elif data_subtype == 'video':
        # Calcular o num_timesteps específico para dados de vídeo
        num_timesteps = calculate_num_timesteps_video(input_data)
    elif data_subtype == 'image':
        # Calcular o num_timesteps específico para dados de imagem
        num_timesteps = calculate_num_timesteps_image(input_data)
    elif data_subtype == 'text':
        # Calcular o num_timesteps específico para dados de texto
        num_timesteps = calculate_num_timesteps_text(input_data)
    elif data_subtype == 'file':
        # Calcular o num_timesteps específico para outros tipos de arquivo
        num_timesteps = calculate_num_timesteps_file(input_data)



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



