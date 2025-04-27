# Chessbot - Système RAG pour l'analyse de positions d'échecs

Ce projet vise à construire un système RAG (Retrieval-Augmented Generation) pour les échecs. L'objectif est de pouvoir interroger une base de données de positions d'échecs vectorisées pour trouver des positions similaires et potentiellement obtenir des analyses ou des informations pertinentes via un modèle de langage augmenté par cette recherche.

## Vue d'ensemble du Workflow

Le cœur du projet réside dans la transformation de parties d'échecs (au format PGN) en représentations vectorielles (embeddings) des positions de l'échiquier (au format FEN). Ces embeddings sont ensuite indexés pour permettre une recherche rapide de similarité.

Le processus se décompose en plusieurs étapes gérées par des scripts dédiés :

1.  **Traitement des fichiers PGN** (`scripts/process_pgn.py`) :
    * Prend en entrée des fichiers PGN situés dans `data/pgn_files/`.
    * Parse chaque partie contenue dans les fichiers PGN.
    * Pour chaque partie, extrait toutes les positions uniques de l'échiquier et les représente au format FEN (Forsyth-Edwards Notation).
    * Collecte également des métadonnées associées à chaque position, comme le coup précédent, le coup suivant, les commentaires éventuels, l'identifiant de la partie, etc..
    * Sauvegarde les FEN et leurs métadonnées dans des fichiers CSV dans `data/processed_files/` (un fichier `*_positions.csv` par PGN traité).
    * Sauvegarde également les métadonnées globales des parties (Event, Site, Date, White, Black, Result, ligne principale des coups avec commentaires) dans des fichiers `*_games.csv`.

2.  **Calcul des Embeddings** (`scripts/calculate_embeddings.py`) :
    * Lit tous les fichiers `*_positions.csv` générés à l'étape précédente.
    * Identifie tous les FEN uniques présents dans ces fichiers.
    * Pour chaque FEN unique qui n'a pas encore d'embedding calculé, appelle une API locale (exposée par `app.py`) qui héberge un modèle pré-entraîné `searchless_chess`.
    * L'API retourne un vecteur d'embedding (une représentation numérique dense) pour le FEN donné. Ce vecteur capture les caractéristiques sémantiques de la position d'échecs selon le modèle `searchless_chess`.
    * Stocke la correspondance FEN <-> Embedding dans un fichier CSV unique : `data/embeddings/fen_embeddings.csv`. Gère la mise à jour incrémentale pour ne calculer que les embeddings manquants.

3.  **Création de l'Index Faiss** (`scripts/create_faiss_index.py`) :
    * Lit le fichier `data/embeddings/fen_embeddings.csv`.
    * Extrait tous les vecteurs d'embeddings.
    * Construit un index Faiss (par exemple, `IndexFlatL2`) à partir de ces embeddings. Faiss est une bibliothèque optimisée pour la recherche de similarités dans de grands ensembles de vecteurs.
    * Sauvegarde l'index Faiss dans `data/faiss/fen_index.faiss`. Cet index permettra de retrouver rapidement les embeddings (et donc les FEN) les plus proches d'un embedding requête.

## Objectif RAG

L'index Faiss créé constitue la base de connaissances vectorielle du système RAG. L'idée est la suivante :

1.  Un utilisateur fournit une position d'échecs (FEN).
2.  Le système calcule l'embedding de cette position requête en utilisant le même modèle `searchless_chess` via l'API.
3.  Le système interroge l'index Faiss avec l'embedding requête pour trouver les 'k' positions les plus similaires (plus proches voisins) dans la base de données.
4.  Les informations associées à ces positions similaires (FEN, commentaires, issue de la partie, etc., potentiellement récupérées depuis les fichiers CSV ou une base de données relationnelle) sont extraites.
5.  Ces informations contextuelles sont injectées dans le prompt d'un modèle de langage (LLM).
6.  Le LLM génère une réponse (analyse, suggestions de coups, explication de thèmes tactiques, etc.) en s'appuyant sur le contexte retrouvé, fournissant ainsi une réponse plus informée et pertinente que s'il se basait uniquement sur ses connaissances internes.

## Utilisation

1.  **Prérequis** :
    * Assurez-vous d'avoir installé les dépendances listées dans `requirements.txt`.
    * Téléchargez et configurez le modèle `searchless_chess` (voir la documentation de `searchless_chess` et le fichier `app.py`).
    * Placez vos fichiers PGN dans le dossier `data/pgn_files/`.

2.  **Lancer l'API d'embedding** :
    ```bash
    docker build -t searchless_chess .
    docker run -d -p 5000:5000 searchless_chess
    ```
    Cette commande démarre le serveur Flask qui expose le modèle `searchless_chess` pour le calcul des embeddings.

3.  **Interroger le système (Futur)** :
    Une fois le pipeline exécuté et l'index créé, un autre composant (non inclus ici) sera nécessaire pour :
    * Accepter une position FEN en entrée.
    * Calculer son embedding via l'API `/embedding`.
    * Interroger l'index Faiss.
    * Construire un prompt avec les résultats.
    * Interagir avec un LLM pour générer la réponse finale.
