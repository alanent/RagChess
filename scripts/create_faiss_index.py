# scripts/create_faiss_index.py
import pandas as pd
import numpy as np
import faiss
import os
import logging
import argparse
import ast # Pour évaluer les embeddings lus comme des chaînes

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Chemins par défaut
DEFAULT_EMBEDDINGS_CSV = "../data/embeddings/fen_embeddings.csv"
DEFAULT_FAISS_INDEX = "../data/faiss/fen_index.faiss"

def load_and_prepare_embeddings(csv_path):
    """Charge les embeddings depuis le CSV et les prépare pour Faiss."""
    if not os.path.exists(csv_path):
        logging.error(f"Le fichier d'embeddings '{csv_path}' n'a pas été trouvé.")
        return None, None

    try:
        df = pd.read_csv(csv_path)
        if 'fen' not in df.columns or 'embedding' not in df.columns:
            logging.error("Le fichier CSV doit contenir les colonnes 'fen' et 'embedding'.")
            return None, None

        # Filtrer les erreurs potentielles stockées comme string
        original_count = len(df)
        df = df[~df['embedding'].astype(str).str.startswith('ERROR_')]
        filtered_count = len(df)
        if original_count > filtered_count:
             logging.warning(f"{original_count - filtered_count} lignes avec erreurs d'embedding ont été ignorées.")

        if df.empty:
            logging.error("Aucun embedding valide trouvé après filtrage.")
            return None, None

        # Convertir la colonne 'embedding' (qui peut être une chaîne) en listes/arrays
        def parse_embedding(emb_str):
            try:
                # Essayer d'évaluer la chaîne comme une liste Python
                return ast.literal_eval(emb_str)
            except (ValueError, SyntaxError, TypeError):
                logging.warning(f"Impossible de parser l'embedding: {emb_str[:100]}... Retourne None.")
                return None # Gérer les cas où ce n'est pas une chaîne de liste valide

        embeddings_list = df['embedding'].apply(parse_embedding).dropna().tolist()

        if not embeddings_list:
            logging.error("Aucun embedding n'a pu être correctement parsé.")
            return None, None

        # Convertir en tableau NumPy float32
        embeddings_np = np.array(embeddings_list).astype('float32')

        # Vérifier la cohérence des dimensions
        if embeddings_np.ndim != 2:
             logging.error(f"Les embeddings n'ont pas la forme attendue (2D). Forme obtenue: {embeddings_np.shape}")
             return None, None

        dimension = embeddings_np.shape[1]
        logging.info(f"Chargé {embeddings_np.shape[0]} embeddings de dimension {dimension}.")

        # Récupérer les FEN correspondants aux embeddings valides
        valid_indices = df['embedding'].apply(parse_embedding).dropna().index
        fens = df.loc[valid_indices, 'fen'].tolist()


        return embeddings_np, fens

    except pd.errors.EmptyDataError:
        logging.error(f"Le fichier CSV '{csv_path}' est vide.")
        return None, None
    except Exception as e:
        logging.error(f"Erreur lors de la lecture ou du traitement de {csv_path}: {e}", exc_info=True)
        return None, None

def create_and_save_faiss_index(embeddings, output_path):
    """Crée un index Faiss et le sauvegarde."""
    if embeddings is None or embeddings.shape[0] == 0:
        logging.error("Aucun embedding fourni pour créer l'index.")
        return False

    dimension = embeddings.shape[1]
    # Utiliser IndexFlatL2 qui est simple et bon pour commencer
    # Pour de très grandes dimensions, IndexFlatIP (produit scalaire) peut être pertinent si normalisé
    index = faiss.IndexFlatL2(dimension)

    try:
        index.add(embeddings)
        logging.info(f"Ajouté {index.ntotal} vecteurs à l'index Faiss.")

        # Créer le dossier de sortie si nécessaire
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Sauvegarder l'index
        faiss.write_index(index, output_path)
        logging.info(f"Index Faiss sauvegardé dans '{output_path}'")
        return True
    except Exception as e:
        logging.error(f"Erreur lors de la création ou sauvegarde de l'index Faiss: {e}", exc_info=True)
        return False

def main():
    parser = argparse.ArgumentParser(description="Crée un index Faiss à partir d'un fichier CSV d'embeddings FEN.")
    parser.add_argument("--embeddings_csv", default=DEFAULT_EMBEDDINGS_CSV,
                        help=f"Chemin vers le fichier CSV des embeddings (défaut: {DEFAULT_EMBEDDINGS_CSV})")
    parser.add_argument("--output_index", default=DEFAULT_FAISS_INDEX,
                        help=f"Chemin où sauvegarder l'index Faiss généré (défaut: {DEFAULT_FAISS_INDEX})")
    args = parser.parse_args()

    logging.info("Début de la création de l'index Faiss.")

    # 1. Charger et préparer les embeddings
    embeddings_np, _ = load_and_prepare_embeddings(args.embeddings_csv) # Ignorer les fens ici

    # 2. Créer et sauvegarder l'index
    if embeddings_np is not None:
        success = create_and_save_faiss_index(embeddings_np, args.output_index)
        if success:
            logging.info("Index Faiss créé avec succès.")
        else:
            logging.error("Échec de la création de l'index Faiss.")
    else:
        logging.error("Impossible de continuer sans embeddings valides.")

    logging.info("Fin du script.")

if __name__ == "__main__":
    main()