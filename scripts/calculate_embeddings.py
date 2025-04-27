# scripts/calculate_embeddings.py
import pandas as pd
import argparse
import os
import glob
import logging
import requests
import json
import time
from tqdm import tqdm # Pour la barre de progression
import ast # Pour évaluer les embeddings lus depuis le CSV si nécessaire

# --- Configuration ---
# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constantes pour l'API (à ajuster selon les besoins)
# METTRE VOTRE VRAIE URL D'API ICI
DEFAULT_API_URL = "http://localhost:5000/embedding" # EXEMPLE: Remplacez par votre URL
# METTRE VOS VRAIS HEADERS ICI (par exemple pour l'authentification)
DEFAULT_HEADERS = {
    'Content-Type': 'application/json',
    # 'Authorization': 'Bearer VOTRE_API_KEY' # Décommentez et adaptez si nécessaire
}
MAX_RETRIES = 3 # Nombre maximum de tentatives pour une requête API échouée
API_RETRY_DELAY = 5 # Délai en secondes entre les tentatives
REQUEST_TIMEOUT = 30 # Timeout en secondes pour la requête API

# Nom du fichier CSV de sortie pour les embeddings
DEFAULT_EMBEDDINGS_FILE = "../data/embeddings/fen_embeddings.csv"
# Répertoire où chercher les fichiers _positions.csv
DEFAULT_POSITIONS_DIR = "../data/processed_files"

# --- Fonction API (fournie dans la question, légèrement adaptée pour clarté) ---
def get_embedding_from_api(fen_string, api_url, headers):
    """Appelle l'API pour obtenir l'embedding d'un FEN, avec gestion des erreurs et re-essais."""
    payload = json.dumps({"fen": fen_string})
    retries = 0

    while retries <= MAX_RETRIES:
        try:
            logging.debug(f"API Call: FEN={fen_string}, URL={api_url}, Tentative={retries+1}")
            response = requests.post(api_url, headers=headers, data=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status() # Lève une exception pour les codes d'erreur HTTP (4xx, 5xx)
            result = response.json()
            logging.debug(f"API Response OK: {result}")

            if 'embedding' in result and isinstance(result['embedding'], list):
                return result['embedding'] # Retourne la liste d'embedding
            else:
                logging.error(f"Format d'embedding invalide pour FEN {fen_string}. Réponse: {result}")
                return "ERROR_INVALID_FORMAT" # Retourne un code d'erreur string

        except requests.exceptions.Timeout:
            logging.warning(f"Timeout pour FEN {fen_string} (Tentative {retries + 1}/{MAX_RETRIES + 1})")
            if retries < MAX_RETRIES:
                logging.info(f"Nouvelle tentative dans {API_RETRY_DELAY} secondes...")
                time.sleep(API_RETRY_DELAY)
                retries += 1
            else:
                logging.error("Nombre maximum de tentatives atteint pour Timeout.")
                return "ERROR_TIMEOUT" # Retourne un code d'erreur string

        except requests.exceptions.RequestException as e:
            status_code = None
            error_response_text = ""
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code
                try:
                    error_response_text = e.response.text # Obtenir le texte brut pour le log
                except Exception:
                    error_response_text = "(Impossible de lire le corps de la réponse)"


            logging.warning(f"Erreur requête API pour FEN {fen_string} (Tentative {retries + 1}/{MAX_RETRIES + 1}), Status: {status_code}, Erreur: {e}")
            if error_response_text:
                 logging.warning(f"Réponse serveur (max 500 chars): {error_response_text[:500]}")

            error_message = "ERROR_REQUEST"
            # Ne pas retenter les erreurs client (4xx) sauf éventuellement 429 (Too Many Requests) si l'API le spécifie
            is_retriable = not (status_code is not None and 400 <= status_code < 500 and status_code != 429)

            if status_code is not None:
                 error_message = f"ERROR_HTTP_{status_code}"

            if is_retriable and retries < MAX_RETRIES:
                logging.info(f"Nouvelle tentative dans {API_RETRY_DELAY} secondes...")
                time.sleep(API_RETRY_DELAY)
                retries += 1
            else:
                logging.error(f"Nombre maximum de tentatives atteint ou erreur non récupérable ({error_message}).")
                return error_message # Retourne un code d'erreur string

        except json.JSONDecodeError:
            logging.error(f"Décodage JSON impossible pour FEN {fen_string}. Réponse serveur probable: {response.text[:200]}...")
            return "ERROR_JSON_DECODE" # Retourne un code d'erreur string

        except Exception as e:
            logging.error(f"Erreur inattendue API pour FEN {fen_string}: {type(e).__name__} - {e}", exc_info=True)
            return "ERROR_UNEXPECTED_API" # Retourne un code d'erreur string

    logging.error(f"Nombre maximum de tentatives dépassé sans succès pour FEN {fen_string}")
    return "ERROR_MAX_RETRIES_EXCEEDED" # Retourne un code d'erreur string


# --- Fonctions Utilitaires ---
def load_existing_embeddings(filepath):
    """Charge les embeddings depuis le fichier CSV s'il existe."""
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            # S'assurer que la colonne fen existe
            if 'fen' not in df.columns:
                logging.warning(f"Le fichier {filepath} ne contient pas de colonne 'fen'. Il sera ignoré/écrasé.")
                return pd.DataFrame(columns=['fen', 'embedding']), set()
             # La colonne 'embedding' est lue comme une chaîne, ce qui est OK pour la sauvegarde.
             # Si vous avez besoin de la relire en tant que liste plus tard, utilisez:
             # df['embedding'] = df['embedding'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
            existing_fens = set(df['fen'].astype(str).unique())
            logging.info(f"Chargé {len(existing_fens)} FENs existants depuis {filepath}")
            return df, existing_fens
        except pd.errors.EmptyDataError:
            logging.warning(f"Le fichier d'embeddings {filepath} est vide. Création d'un nouveau fichier.")
            return pd.DataFrame(columns=['fen', 'embedding']), set()
        except Exception as e:
            logging.error(f"Erreur lors de la lecture de {filepath}: {e}. Création d'un nouveau fichier.", exc_info=True)
            # En cas d'erreur de lecture, mieux vaut recommencer pour éviter la corruption
            return pd.DataFrame(columns=['fen', 'embedding']), set()
    else:
        logging.info(f"Le fichier d'embeddings {filepath} n'existe pas. Il sera créé.")
        return pd.DataFrame(columns=['fen', 'embedding']), set()

def find_position_files(directory):
    """Trouve tous les fichiers *_positions.csv dans le répertoire donné."""
    pattern = os.path.join(directory, "*_positions.csv")
    files = glob.glob(pattern)
    logging.info(f"Trouvé {len(files)} fichier(s) *_positions.csv dans {directory}")
    return files

def extract_unique_fens_from_files(file_list):
    """Extrait les FENs uniques de tous les fichiers CSV fournis."""
    all_fens = set()
    total_rows = 0
    if not file_list:
        return set()

    logging.info("Lecture des FENs depuis les fichiers _positions.csv...")
    for filepath in tqdm(file_list, desc="Lecture des fichiers FEN"):
        try:
            df_pos = pd.read_csv(filepath)
            if 'fen' in df_pos.columns:
                # Utiliser .astype(str) au cas où certains FEN seraient interprétés comme autre chose
                unique_fens_in_file = set(df_pos['fen'].astype(str).unique())
                all_fens.update(unique_fens_in_file)
                total_rows += len(df_pos)
            else:
                logging.warning(f"Le fichier {os.path.basename(filepath)} n'a pas de colonne 'fen'. Ignoré.")
        except pd.errors.EmptyDataError:
             logging.warning(f"Le fichier {os.path.basename(filepath)} est vide. Ignoré.")
        except Exception as e:
            logging.error(f"Erreur lors de la lecture de {filepath}: {e}", exc_info=True)

    logging.info(f"Lu {total_rows} positions au total. Trouvé {len(all_fens)} FENs uniques.")
    return all_fens

def save_embeddings(df, filepath):
    """Sauvegarde le DataFrame des embeddings dans un fichier CSV."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False, encoding='utf-8')
        logging.info(f"Sauvegardé {len(df)} entrées FEN/embedding dans {filepath}")
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde des embeddings dans {filepath}: {e}", exc_info=True)

# --- Fonction Principale ---
def main():
    parser = argparse.ArgumentParser(description="Calcule les embeddings pour les FENs uniques trouvés dans les fichiers *_positions.csv.")
    parser.add_argument("--positions_dir", default=DEFAULT_POSITIONS_DIR,
                        help=f"Dossier contenant les fichiers *_positions.csv (défaut: {DEFAULT_POSITIONS_DIR})")
    parser.add_argument("--embeddings_file", default=DEFAULT_EMBEDDINGS_FILE,
                        help=f"Fichier CSV où sauvegarder/mettre à jour les embeddings (défaut: {DEFAULT_EMBEDDINGS_FILE})")
    parser.add_argument("--api_url", default=DEFAULT_API_URL,
                        help=f"URL de l'API d'embedding (défaut: {DEFAULT_API_URL})")
    parser.add_argument("--api_key", default=None,
                        help="Clé API (Bearer Token) si nécessaire pour l'authentification (optionnel).")
    args = parser.parse_args()

    # Vérifier le dossier d'entrée
    if not os.path.isdir(args.positions_dir):
        logging.error(f"Le dossier des positions '{args.positions_dir}' n'existe pas.")
        return

    # Préparer les headers de l'API
    api_headers = DEFAULT_HEADERS.copy()
    if args.api_key:
        api_headers['Authorization'] = f'Bearer {args.api_key}'
        logging.info("Utilisation d'une clé API pour l'authentification.")

    # 1. Charger les embeddings existants
    df_embeddings, existing_fens = load_existing_embeddings(args.embeddings_file)

    # 2. Trouver les fichiers de positions
    position_files = find_position_files(args.positions_dir)
    if not position_files:
        logging.warning("Aucun fichier *_positions.csv trouvé. Aucune nouvelle FEN à traiter.")
        # Sauvegarder quand même au cas où le fichier initial était corrompu et a été vidé
        save_embeddings(df_embeddings, args.embeddings_file)
        return

    # 3. Extraire tous les FENs uniques des fichiers trouvés
    all_fens_in_files = extract_unique_fens_from_files(position_files)
    if not all_fens_in_files:
        logging.warning("Aucun FEN trouvé dans les fichiers _positions.csv.")
        save_embeddings(df_embeddings, args.embeddings_file)
        return

    # 4. Identifier les FENs qui nécessitent un calcul d'embedding
    fens_to_process = all_fens_in_files - existing_fens
    logging.info(f"Total FENs uniques trouvés: {len(all_fens_in_files)}")
    logging.info(f"FENs déjà présents dans {os.path.basename(args.embeddings_file)}: {len(existing_fens)}")
    logging.info(f"Nouveaux FENs à traiter: {len(fens_to_process)}")

    # 5. Calculer les embeddings pour les nouveaux FENs
    new_embeddings_data = []
    if fens_to_process:
        logging.info(f"Appel de l'API pour {len(fens_to_process)} nouveaux FENs...")
        api_errors = 0
        for fen in tqdm(fens_to_process, desc="Calcul des embeddings"):
            embedding_result = get_embedding_from_api(fen, args.api_url, api_headers)
            # Si l'API renvoie une erreur (string), on la stocke telle quelle
            if isinstance(embedding_result, str) and embedding_result.startswith("ERROR_"):
                 api_errors += 1
                 logging.warning(f"Erreur API pour FEN {fen}: {embedding_result}")
            elif not isinstance(embedding_result, list):
                 # Sécurité supplémentaire au cas où l'API renverrait qqch d'inattendu non-liste
                 api_errors += 1
                 logging.error(f"Type de retour inattendu de l'API pour FEN {fen}: {type(embedding_result)}")
                 embedding_result = "ERROR_UNEXPECTED_TYPE"

            new_embeddings_data.append({'fen': fen, 'embedding': embedding_result})

        logging.info(f"Calcul des embeddings terminé. {len(new_embeddings_data) - api_errors} succès, {api_errors} erreurs API.")

        # 6. Préparer les nouvelles données à ajouter
        if new_embeddings_data:
            df_new = pd.DataFrame(new_embeddings_data)
            # Convertir la liste d'embedding en chaîne pour le CSV (comportement par défaut de to_csv)
            # df_new['embedding'] = df_new['embedding'].astype(str)

            # 7. Combiner les anciens et nouveaux embeddings
            # Utiliser ignore_index=True pour réinitialiser l'index du DataFrame combiné
            df_combined = pd.concat([df_embeddings, df_new], ignore_index=True)

            # Optionnel : Supprimer les doublons de FEN au cas où (ne devrait pas arriver avec la logique actuelle, mais par sécurité)
            # Garde la première occurrence (normalement l'ancienne si elle existait)
            # df_combined = df_combined.drop_duplicates(subset=['fen'], keep='first')

            # Remplacer le DataFrame principal par le combiné pour la sauvegarde
            df_embeddings = df_combined

    else:
        logging.info("Aucun nouveau FEN à traiter via l'API.")

    # 8. Sauvegarder le fichier CSV complet (écrase l'ancien)
    # S'assurer que les colonnes sont dans le bon ordre
    if not df_embeddings.empty:
        df_embeddings = df_embeddings[['fen', 'embedding']] # Réordonne si nécessaire
    save_embeddings(df_embeddings, args.embeddings_file)

    logging.info("Traitement terminé.")

if __name__ == "__main__":
    main()