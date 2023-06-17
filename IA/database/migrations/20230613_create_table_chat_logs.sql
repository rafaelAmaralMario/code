-- Criação da tabela ChatLogs
CREATE TABLE chat_logs (
    id SERIAL PRIMARY KEY,
    user_message TEXT,
    ai_response TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
