# Imagem base
FROM python:3.9

# Diretório de trabalho
WORKDIR /app

# Copiar o arquivo de requisitos
COPY requirements.txt .

# Instalar as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o restante dos arquivos
COPY . .

# Definir o comando de inicialização
CMD [ "python", "main.py" ]
