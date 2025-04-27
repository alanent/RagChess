# Use a Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (git for cloning, graphviz for the original setup)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Clone the repository (alternatively, copy it if you clone it outside first)
RUN git clone https://github.com/google-deepmind/searchless_chess.git ./searchless_chess

# Set the working directory to inside the cloned repo for requirements installation
WORKDIR /app/searchless_chess

# [...] Autres instructions précédentes

# Install Python dependencies (use the requirements file from the parent directory)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# --- MODIFICATION ICI ---
# Supprimer ou commenter la ligne de téléchargement :
# RUN ./checkpoints/download.sh ${MODEL_NAME}

# Vérifier que le répertoire checkpoints existe dans le WORKDIR courant (/app/searchless_chess)
RUN mkdir -p ./checkpoints/

# Copier le répertoire de checkpoints local (contenant 270M) dans l'image
# Assurez-vous que searchless_chess/checkpoints/270M existe localement par rapport au Dockerfile
ARG MODEL_NAME=270M 
COPY searchless_chess/checkpoints/${MODEL_NAME} ./checkpoints/${MODEL_NAME}/
# --- FIN DE LA MODIFICATION ---


# Copier votre code d'application API
WORKDIR /app
COPY app.py .

# --- Configuration ---
ENV MODEL_NAME=${MODEL_NAME}
ENV POLICY=action_value 
ENV NUM_RETURN_BUCKETS=128
ENV PORT=5000

# Expose the port the app runs on
EXPOSE ${PORT}

# Command to run the application
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]