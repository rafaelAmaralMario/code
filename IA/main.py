import os
import nltk
import numpy as np
import neural_networks
from flask import Flask, jsonify, request
from keras.models import model_from_json
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType, StructField, StructType
import uuid
from utils import get_data_type, get_numtime_steps,get_extracted_data


nltk.download('punkt')

app = Flask(__name__)

num_models = 5

# Inicialização do SparkSession
spark = SparkSession.builder.master("local[*]").appName("AIPrototype").getOrCreate()


# Definição do esquema da tabela TrainingData
training_data_schema = StructType([
    StructField("id", IntegerType(), nullable=False),
    StructField("input_data", StringType(), nullable=True),
    StructField("target_data", StringType(), nullable=True)
])

# Definição do esquema da tabela ChatLog
chat_log_schema = StructType([
    StructField("user_message", StringType(), nullable=True),
    StructField("ai_response", StringType(), nullable=True)
])

# Definição do esquema da tabela trained_models
trained_models_schema = StructType([
    StructField("id", IntegerType(), nullable=False),
    StructField("name", StringType(), nullable=False),
    StructField("file_content", StringType(), nullable=False)
])


# Funções auxiliares para manipulação dos dados

def save_training_data_to_spark(input_data, target_data):
    # Prepara os dados para salvar no Spark DataFrame
    data = [(i, input_val, target_val) for i, (input_val, target_val) in enumerate(zip(input_data, target_data))]
    df = spark.createDataFrame(data, schema=training_data_schema)

    # Salva os dados na tabela TrainingData
    df.write.mode("append").saveAsTable("TrainingData")

def save_chat_log_to_spark(user_message, ai_response):
    # Prepara os dados para salvar no Spark DataFrame
    chat_log = [(user_message, ai_response)]
    df = spark.createDataFrame(chat_log, schema=chat_log_schema)

    # Salva o log na tabela ChatLog
    df.write.mode("append").format("parquet").saveAsTable("ChatLog")


def save_model_to_spark(model_name, file_content):

    model_id = str(uuid.uuid4())
    model_name = model_id
    # Prepara os dados para salvar no Spark DataFrame
    data = [(model_id, model_name, file_content)]
    df = spark.createDataFrame(data, ["id", "name", "file_content"])

    # Salva os dados na tabela trained_models
    df.write.mode("append").saveAsTable("trained_models")

def load_models_from_spark():
    # Carrega os dados da tabela trained_models
    df = spark.table("trained_models")

    # Converte os dados do DataFrame para uma lista de dicionários
    models = [
        {
            "id": row.id,
            "name": row.name,
            "file_content": row.file_content
        }
        for row in df.collect()
    ]

    return models



def load_training_data_from_spark():
    # Carrega os dados da tabela TrainingData
    df = spark.table("TrainingData")

    # Converte os dados do DataFrame para listas
    input_data = df.select("input_data").rdd.flatMap(lambda x: x).collect()
    target_data = df.select("target_data").rdd.flatMap(lambda x: x).collect()

    return input_data, target_data

def load_chat_log_from_spark():
    # Carrega o log da tabela ChatLog
    df = spark.table("ChatLog")

    # Converte os dados do DataFrame para uma lista de dicionários
    chat_log = df.rdd.map(lambda row: {"user_message": row.user_message, "ai_response": row.ai_response}).collect()

    return chat_log

# Rotas da aplicação Flask

@app.route('/data', methods=['GET'])
def get_data():
    input_data, target_data = load_training_data_from_spark()
    data = [{"input_data": input_val, "target_data": target_val} for input_val, target_val in zip(input_data, target_data)]
    return jsonify(data=data)

@app.route('/train', methods=['POST'])
def train_model():
    if request.headers['Content-Type'] != 'multipart/form-data':
        return 'Unsupported media type. Please use multipart/form-data.', 415
    
    input_data = request.files['input_data']
    name = request.form['name']
    target_data = request.form['target_data']
    save_training_data_to_spark(input_data, target_data)

    models = neural_networks.create_neural_networks(input_data, num_models, True)
    trained_models = train_models(input_data, target_data, models)
    save_model_to_spark(name,trained_models)
    save_models_locally(name,trained_models)
    
    return jsonify(message='Training completed')

@app.route('/predict', methods=['POST'])
def predict_model():
    input_data = request.files['input_data']
    models = load_models_locally()
    predictions = predict_models(input_data, models)
    combined_predictions = combine_predictions(predictions)
    processed_predictions = postprocess(combined_predictions)
    return jsonify(predictions=processed_predictions)

@app.route('/message', methods=['POST'])
def process_message():
    user_message = request.json['message']
    models = neural_networks.create_neural_networks(user_message, num_models, False)
    trained_models = train_unsupervised_models(user_message, models)
    save_models_locally(trained_models)
    ai_response = generate_ai_response(user_message)
    save_chat_log_to_spark(user_message,ai_response)
    return jsonify(response=ai_response)

@app.route('/message', methods=['GET'])
def get_messages():
    history = load_chat_log_from_spark()
    return jsonify(response=history)

@app.route('/train_unsupervised', methods=['POST'])
def train_unsupervised():
    if request.mimetype != 'multipart/form-data':
        return 'Unsupported media type. Please use multipart/form-data.', 415
    
    return train_unsupervised_model()

# Resto do código

def train_unsupervised_model():
    input_data = request.files['input_data']
     # Acessar os dados do arquivo
    file_data = input_data.stream.read()
    name = request.form['name']
    filename = input_data.filename

    data_type, data_subtype = get_data_type(filename)

    extracted_data = get_extracted_data(file_data, data_subtype)
    file_size = file_data.__len__()
    
    timestep_size = {
    'audio': 0.001,   # 1 millisecond
    'video': 0.02,    # 20 milliseconds
    'image': 0.05,    # 50 milliseconds
    'text': 0.5,      # 500 milliseconds (0.5 seconds)
    'pdf': 0.5,       # 500 milliseconds (0.5 seconds)
    'csv': 0.05,      # 50 milliseconds
    'excel': 0.05,    # 50 milliseconds
    'word': 0.05      # 50 milliseconds
    }
    
    num_timesteps = int(file_size / timestep_size.get(data_subtype))

    models = neural_networks.create_neural_networks(5,False,num_timesteps, data_type, file_size)

    unsupervised_models = train_unsupervised_models(extracted_data, models)

    save_model_to_spark(name,unsupervised_models)
    save_models_locally(name,unsupervised_models)

    return jsonify(message='Unsupervised training completed')

def generate_data(input_data):
    generated_data = []
    for data in input_data:
        noisy_data = add_noise(data)
        generated_data.append(noisy_data)
    return np.array(generated_data)

def add_noise(data):
    noise = np.random.normal(0, 0.1, data.shape)
    noisy_data = data + noise
    return noisy_data

def train_unsupervised_models(input_data, models):
    unsupervised_models = []
    for model in models:
        generated_data = generate_data(input_data)
        preprocessed_data = preprocess(generated_data)
        model.fit(preprocessed_data, preprocessed_data, epochs=10, batch_size=32)
        unsupervised_models.append(model)
    return unsupervised_models

# Função para aplicação da função de ativação
def apply_activation_function(value):
    activated_value = sigmoid(value)
    return activated_value

# Função para pré-processamento dos dados
def preprocess_data(data):
    normalized_data = normalize_data(data)
    return normalized_data

# Função sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Função de normalização dos dados
def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / std
    return normalized_data

def train_models(input_data, target_data, models):
    trained_models = []
    for model in models:
        model.fit(input_data, target_data, epochs=10, batch_size=32)
        trained_models.append(model)

    return trained_models

def save_models_locally(name,models):
    if not os.path.exists("models"):
        os.makedirs("models")

    for idx, model in enumerate(models):
        serialized_model = model.to_json()
        model_path = f"models/model_{name}.txt"
        with open(model_path, "w") as file:
            file.write(serialized_model)

def load_models_locally():
    models = []
    if not os.path.exists("models"):
        return models

    model_files = os.listdir("models")
    for file_name in model_files:
        model_path = os.path.join("models", file_name)
        with open(model_path, "r") as file:
            serialized_model = file.read()
            model = model_from_json(serialized_model)
            models.append(model)

    return models

def predict_models(input_data, models):
    predictions = []
    for model in models:
        prediction = model.predict(input_data)
        predictions.append(prediction)

    return predictions

# Função para combinar as predições dos modelos
def combine_predictions(predictions):
    combined_predictions = []
    for i in range(len(predictions[0])):
        combined_prediction = []
        for j in range(len(predictions)):
            combined_prediction.append(predictions[j][i])
        combined_predictions.append(combined_prediction)
    return combined_predictions

# Função para pré-processamento dos dados
def preprocess(input_data):
    preprocessed_data = []
    for data in input_data:
        preprocessed_value = preprocess_data(data)
        preprocessed_data.append(preprocessed_value)
    return preprocessed_data

def postprocess(predictions):
    processed_predictions = []
    for prediction in predictions:
        processed_prediction = []
        for value in prediction:
            processed_value = apply_activation_function(value)
            processed_prediction.append(processed_value)
        processed_predictions.append(processed_prediction)
    return processed_predictions

def generate_ai_response(user_message):
    models = load_models_locally()
    preprocessed_message = preprocess([user_message])
    predictions = predict_models(preprocessed_message, models)
    combined_predictions = combine_predictions(predictions)
    ai_response = postprocess(combined_predictions)
    return ai_response[0]  # Assume que haverá apenas uma resposta

if __name__ == '__main__':
    app.run(debug=True)
