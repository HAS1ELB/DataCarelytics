#!/bin/bash

# Vérification de l'existence du fichier .env
if [ ! -f .env ]; then
    echo "Le fichier .env n'existe pas."
    echo "Veuillez créer un fichier .env basé sur .env-example et définir vos variables d'environnement."
    exit 1
fi

# Démarrage de l'application avec Docker Compose
docker-compose up -d

echo "DataCarelytics est en cours de démarrage..."
echo "Accédez à l'application à l'adresse: http://localhost:8501"
