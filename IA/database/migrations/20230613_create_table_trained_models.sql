-- Criação da tabela TrainedModels
CREATE TABLE trained_models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    file_content TEXT
);
