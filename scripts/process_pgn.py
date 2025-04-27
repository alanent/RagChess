# scripts/process_pgn.py
import chess.pgn
import pandas as pd
import argparse
import os
import logging
import uuid
import glob
import chess # Assurez-vous que chess est importé

# Configuration du logging (inchangé)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# generate_game_name (inchangé)
def generate_game_name(game_headers):
    # ... (code identique) ...
    try:
        site = game_headers.get('Site', '?')
        date = game_headers.get('Date', '?').replace('.', '-')
        white = game_headers.get('White', 'W?')
        black = game_headers.get('Black', 'B?')
        round_info = game_headers.get('Round', 'R?')
        game_name_str = f"{site}_{date}_{round_info}_{white}_vs_{black}"
        # Sanitize filename characters
        game_name_str = "".join(c if c.isalnum() or c in ['-', '_', ' '] else '_' for c in game_name_str)
        return game_name_str[:150] # Limit length
    except Exception as e:
        logging.warning(f"Erreur génération nom partie: {e}. Utilisation de 'Unknown_Game_Name'.")
        return "Unknown_Game_Name"


def extract_fen_and_comments(pgn_file_path):
    """
    Extrait les détails des positions, les métadonnées (en-têtes) ET la liste
    des coups/commentaires de la ligne principale pour chaque partie.

    Returns:
        tuple(list, list): Un tuple contenant deux listes:
                           1. Liste des dictionnaires de positions.
                           2. Liste des dictionnaires de métadonnées de parties,
                              incluant une chaîne 'mainline_moves_comments'.
    """
    positions_data = []
    games_metadata = []
    game_count = 0
    processed_positions_count = 0

    try:
        with open(pgn_file_path, 'r', encoding='utf-8', errors='replace') as pgn:
            while True:
                try:
                    game = chess.pgn.read_game(pgn)
                except Exception as read_err:
                    logging.warning(f"Erreur lecture entête/partie dans {os.path.basename(pgn_file_path)}: {read_err}. Passage.")
                    game = None

                if game is None:
                    break

                game_count += 1
                game_id = str(uuid.uuid4())
                game_name = generate_game_name(game.headers)

                # --- Initialisation pour cette partie ---
                board = game.board()
                mainline_data = [] # Pour stocker les infos de la ligne principale (san, comment)

                # --- Traitement des positions ET collecte des coups/commentaires ---
                # Position initiale (avant le premier coup)
                initial_fen = board.fen()
                initial_comment_on_node = game.comment.strip() if game.comment else "" # Commentaire AVANT le 1er coup
                initial_last_move = ""
                initial_next_move_san = ""
                if game.variations:
                    try:
                        first_move = game.variation(0).move
                        initial_next_move_san = board.san(first_move)
                    except Exception as san_err:
                         logging.warning(f"Erreur SAN pour 1er coup partie {game_id}: {san_err}")
                         try: initial_next_move_san = str(game.variation(0).move)
                         except: pass

                positions_data.append({
                    'game_id': game_id, 'game_name': game_name, 'ply_number': 0,
                    'fen': initial_fen, 'comment': initial_comment_on_node, # Commentaire sur le noeud racine
                    'last_move_san': initial_last_move, 'next_move_san': initial_next_move_san
                })
                processed_positions_count += 1

                # Itération sur les coups de la ligne principale
                ply_counter = 0
                node = game
                last_move_san_for_current_fen = ""

                while node.variations:
                    try:
                        next_node = node.variation(0)
                        move = next_node.move
                        # Commentaire associé à CE coup/noeud
                        comment_on_move = next_node.comment.strip() if next_node.comment else ""

                        # Obtenir le SAN du coup courant (avant de le jouer sur le board)
                        try:
                            current_move_san = board.san(move)
                        except Exception as san_err:
                            logging.warning(f"Erreur SAN (current) partie {game_id} ply {ply_counter+1}: {san_err}")
                            current_move_san = str(move) # Fallback UCI

                        # --- Stocker le coup et son commentaire pour la liste finale ---
                        mainline_data.append({"san": current_move_san, "comment": comment_on_move, "turn": board.turn})

                        # Appliquer le coup pour la suite et l'analyse FEN
                        board.push(move)
                        ply_counter += 1
                        current_fen = board.fen()
                        last_move_san_for_current_fen = current_move_san # Mettre à jour pour le FEN généré

                        # Trouver le prochain coup (pour l'enregistrement FEN)
                        next_move_san_from_current_fen = ""
                        if next_node.variations:
                            try:
                                next_next_move = next_node.variation(0).move
                                next_move_san_from_current_fen = board.san(next_next_move)
                            except Exception as san_err:
                                logging.warning(f"Erreur SAN (next) partie {game_id} ply {ply_counter}: {san_err}")
                                try: next_move_san_from_current_fen = str(next_node.variation(0).move)
                                except: pass

                        # Enregistrer les données de la position FEN
                        positions_data.append({
                            'game_id': game_id, 'game_name': game_name, 'ply_number': ply_counter,
                            'fen': current_fen, 'comment': comment_on_move, # Commentaire associé au coup menant à ce FEN
                            'last_move_san': last_move_san_for_current_fen,
                            'next_move_san': next_move_san_from_current_fen
                        })
                        processed_positions_count += 1
                        node = next_node # Avancer dans la ligne principale

                    except Exception as move_err:
                        logging.warning(f"Erreur traitement coup/commentaire partie {game_id} (partie {game_count}) "
                                        f"fichier {os.path.basename(pgn_file_path)} après ply {ply_counter}: {move_err}. "
                                        f"Fin traitement positions/coups pour cette partie.")
                        break # Arrêter le traitement de cette partie

                # --- Fin de la partie : Formater la liste des coups/commentaires ---
                formatted_moves_list = []
                fullmove_num = 1
                for i, move_info in enumerate(mainline_data):
                    san = move_info["san"]
                    comment = move_info["comment"]
                    turn = move_info["turn"] # Qui allait jouer avant ce coup ?

                    move_str = ""
                    # Ajouter le numéro de coup pour les Blancs
                    if turn == chess.WHITE:
                        move_str += f"{fullmove_num}. "

                    move_str += san

                    if comment:
                        move_str += f" {{{comment}}}"

                    formatted_moves_list.append(move_str)

                    # Incrémenter le numéro après le coup des Noirs
                    if turn == chess.BLACK:
                        fullmove_num += 1

                final_moves_string = " ".join(formatted_moves_list)
                # Ajouter aussi le résultat à la fin si disponible dans les headers?
                game_result = game.headers.get("Result", "")
                if final_moves_string and game_result != "*": # N'ajoute pas le résultat si la partie est vide ou non terminée
                     final_moves_string += f" {game_result}"


                # --- Stocker les METADONNEES enrichies de la partie ---
                headers = game.headers
                game_meta_entry = {
                    'game_id': game_id,
                    'event': headers.get('Event', '?'),
                    'site': headers.get('Site', '?'),
                    'date': headers.get('Date', '?'),
                    'round': headers.get('Round', '?'),
                    'white': headers.get('White', '?'),
                    'black': headers.get('Black', '?'),
                    'result': game_result, # Utiliser la variable déjà récupérée
                    'mainline_moves_comments': final_moves_string # Nouvelle colonne
                    # Ajoutez d'autres en-têtes si nécessaire
                }
                games_metadata.append(game_meta_entry)


            logging.info(f"Extrait {processed_positions_count} positions et {len(games_metadata)} entrées de métadonnées (avec coups/comm.) depuis {game_count} parties dans {os.path.basename(pgn_file_path)}")
            if game_count == 0:
                logging.warning(f"Aucune partie valide trouvée dans {pgn_file_path}")
            return positions_data, games_metadata

    except FileNotFoundError:
        logging.error(f"Erreur: Fichier PGN '{pgn_file_path}' non trouvé.")
        return None, None
    except Exception as e:
        logging.error(f"Erreur inattendue majeure lors de la lecture de {pgn_file_path}: {e}", exc_info=True)
        return positions_data, games_metadata


# save_positions_to_csv (inchangé)
def save_positions_to_csv(data, filename):
    # ... (code identique) ...
    if data:
        df = pd.DataFrame(data, columns=['game_id', 'game_name', 'ply_number', 'fen',
                                         'comment', 'last_move_san', 'next_move_san'])
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            df.to_csv(filename, index=False, encoding='utf-8')
            logging.info(f"{len(data)} enregistrements de positions sauvegardés dans {filename}")
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde des positions dans {filename}: {e}")
    else:
        pass


# --- Fonction de sauvegarde MODIFIEE pour les METADONNEES enrichies ---
def save_games_metadata_to_csv(data, filename):
    """
    Sauvegarde la liste des métadonnées des parties (incluant la ligne principale)
    dans un fichier CSV.
    """
    if data:
        # Ajouter la nouvelle colonne
        columns = ['game_id', 'event', 'site', 'date', 'round', 'white', 'black', 'result',
                   'mainline_moves_comments'] # Nouvelle colonne ajoutée
        df = pd.DataFrame(data, columns=columns)
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            df.to_csv(filename, index=False, encoding='utf-8')
            logging.info(f"{len(data)} enregistrements de métadonnées de parties (avec coups/comm.) sauvegardés dans {filename}")
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde des métadonnées de parties dans {filename}: {e}")
    else:
        pass


def main():
    # Parser (inchangé)
    parser = argparse.ArgumentParser(description="Extrait positions FEN, métadonnées de parties et ligne principale (coups+commentaires) de fichiers PGN vers des fichiers CSV.")
    parser.add_argument("--pgn_dir", default="../data/pgn_files",
                        help="Dossier contenant les fichiers PGN à traiter.")
    parser.add_argument("--output_dir", default="../data/processed_files",
                        help="Dossier où sauvegarder les fichiers CSV.")
    args = parser.parse_args()

    # Vérifications initiales et création dossier sortie (inchangé)
    if not os.path.isdir(args.pgn_dir):
        logging.error(f"Le dossier PGN '{args.pgn_dir}' n'existe pas.")
        return
    os.makedirs(args.output_dir, exist_ok=True)
    pgn_files = glob.glob(os.path.join(args.pgn_dir, "*.pgn"))
    if not pgn_files:
        logging.warning(f"Aucun fichier .pgn trouvé dans '{args.pgn_dir}'.")
        return
    logging.info(f"Trouvé {len(pgn_files)} fichier(s) PGN à traiter.")

    processed_count = 0
    skipped_count = 0
    error_count = 0

    for pgn_file_path in pgn_files:
        base_filename = os.path.splitext(os.path.basename(pgn_file_path))[0]
        # Garder le nommage précédent, mais le contenu a changé
        output_positions_csv_path = os.path.join(args.output_dir, f"{base_filename}_positions.csv")
        output_games_csv_path = os.path.join(args.output_dir, f"{base_filename}_games.csv")

        # Logique de Skip (inchangée)
        if os.path.exists(output_positions_csv_path) and os.path.exists(output_games_csv_path):
            logging.info(f"Fichiers de sortie '{os.path.basename(output_positions_csv_path)}' et "
                         f"'{os.path.basename(output_games_csv_path)}' existent déjà. Ignoré.")
            skipped_count += 1
            continue

        logging.info(f"Traitement : {os.path.basename(pgn_file_path)}")
        positions_data = None
        games_metadata = None
        try:
            # Récupérer les données (inclut maintenant la ligne principale formatée)
            positions_data, games_metadata = extract_fen_and_comments(pgn_file_path)

            # Sauvegarder les positions (inchangé)
            if positions_data is not None:
                 if not os.path.exists(output_positions_csv_path):
                     if positions_data:
                         save_positions_to_csv(positions_data, output_positions_csv_path)

            # Sauvegarder les métadonnées enrichies
            if games_metadata is not None:
                if not os.path.exists(output_games_csv_path):
                    if games_metadata:
                        save_games_metadata_to_csv(games_metadata, output_games_csv_path)
                    else:
                         if not positions_data:
                             logging.warning(f"Aucune donnée (position ou métadonnée) extraite de {os.path.basename(pgn_file_path)}. Aucun fichier CSV créé.")
                             skipped_count += 1

            # Compter comme traité (inchangé)
            if (positions_data is not None or games_metadata is not None) and (positions_data or games_metadata):
                 processed_count += 1
            elif positions_data is None and games_metadata is None:
                 error_count += 1

        except Exception as e:
            # Gestion d'erreur majeure (inchangée)
            logging.error(f"Erreur majeure lors du traitement de {os.path.basename(pgn_file_path)}: {e}", exc_info=True)
            error_count += 1
            for path in [output_positions_csv_path, output_games_csv_path]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                        logging.info(f"Supprimé le fichier CSV potentiellement incomplet suite à une erreur: {path}")
                    except OSError as rm_err:
                        logging.warning(f"Impossible de supprimer le fichier CSV incomplet {path}: {rm_err}")

    logging.info(f"Traitement de tous les fichiers terminé.")
    logging.info(f"Résultat: {processed_count} PGN traités (données extraites), {skipped_count} ignorés (déjà traités ou vides/invalides), {error_count} erreurs.")
    logging.info(f"Les détails des positions sont dans '{args.output_dir}/*_positions.csv'.")
    # Log mis à jour pour le deuxième fichier
    logging.info(f"Les métadonnées des parties (avec coups/commentaires) sont dans '{args.output_dir}/*_games.csv'.")

if __name__ == "__main__":
    main()