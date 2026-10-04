# FinanceAgent — image FastAPI seule.
# Ollama N'EST PAS dans ce conteneur : il reste sur la machine hôte
# (choix délibéré — voir docker-compose.yml, variable OLLAMA_BASE_URL).

FROM python:3.12-slim

WORKDIR /app

# Installer les dépendances d'abord (étape mise en cache par Docker tant que
# requirements.txt ne change pas — accélère les reconstructions)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
