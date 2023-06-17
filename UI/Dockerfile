# Imagem base
FROM node:14-alpine

# Diretório de trabalho
WORKDIR /app

# Copiar o arquivo package.json
COPY package.json .

# Instalar as dependências
RUN npm install

# Copiar o restante dos arquivos
COPY . .

# Compilar os arquivos da UI
RUN npm run build

# Definir o comando de inicialização
CMD [ "npm", "start" ]
