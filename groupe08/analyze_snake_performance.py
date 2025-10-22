#!/usr/bin/env python3
"""
Analyseur de performance pour le jeu Snake avec IA BFS
Génère des graphiques à partir des données JSON collectées pendant le jeu
"""

import json
import matplotlib
matplotlib.use('Agg')  # Utiliser le backend non-interactif
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def load_game_data(filename="snake_analysis.json"):
    """Charge les données du fichier JSON."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Erreur: Le fichier {filename} n'existe pas.")
        return None
    except json.JSONDecodeError:
        print(f"Erreur: Le fichier {filename} n'est pas un JSON valide.")
        return None

def plot_score_timeline(data, output_dir="plots"):
    """Génère un graphique du score en fonction du temps."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    times = [point['time'] for point in data['score_timeline']]
    scores = [point['score'] for point in data['score_timeline']]
    
    plt.figure(figsize=(12, 6))
    plt.plot(times, scores, 'b-', linewidth=2, marker='o', markersize=3)
    plt.title(f'Évolution du Score - Algorithme {data["algorithm"]}', fontsize=16)
    plt.xlabel('Temps (secondes)', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Ajouter des informations
    plt.text(0.02, 0.98, f'Score final: {data["final_score"]}\nTemps total: {data["final_time"]}s\nRésultat: {data["game_result"]}', 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/score_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()  # Fermer la figure au lieu de l'afficher

def plot_snake_growth(data, output_dir="plots"):
    """Génère un graphique de la croissance du serpent."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    times = [point['time'] for point in data['score_timeline']]
    lengths = [point['snake_length'] for point in data['score_timeline']]
    
    plt.figure(figsize=(12, 6))
    plt.plot(times, lengths, 'g-', linewidth=2, marker='s', markersize=3)
    plt.title(f'Croissance du Serpent - Algorithme {data["algorithm"]}', fontsize=16)
    plt.xlabel('Temps (secondes)', fontsize=12)
    plt.ylabel('Longueur du Serpent', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Ligne de référence pour la longueur initiale
    plt.axhline(y=3, color='r', linestyle='--', alpha=0.7, label='Longueur initiale')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/snake_growth.png', dpi=300, bbox_inches='tight')
    plt.close()  # Fermer la figure au lieu de l'afficher

def plot_fill_rate(data, output_dir="plots"):
    """Génère un graphique du taux de remplissage de la grille."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    times = [point['time'] for point in data['score_timeline']]
    fill_rates = [point['fill_rate'] for point in data['score_timeline']]
    
    plt.figure(figsize=(12, 6))
    plt.plot(times, fill_rates, 'purple', linewidth=2, marker='^', markersize=3)
    plt.title(f'Taux de Remplissage de la Grille - Algorithme {data["algorithm"]}', fontsize=16)
    plt.xlabel('Temps (secondes)', fontsize=12)
    plt.ylabel('Taux de Remplissage (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Ligne de référence pour 100%
    plt.axhline(y=100, color='r', linestyle='--', alpha=0.7, label='Grille complète (100%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fill_rate.png', dpi=300, bbox_inches='tight')
    plt.close()  # Fermer la figure au lieu de l'afficher

def plot_performance_metrics(data, output_dir="plots"):
    """Génère un graphique combiné avec plusieurs métriques."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    times = [point['time'] for point in data['score_timeline']]
    scores = [point['score'] for point in data['score_timeline']]
    lengths = [point['snake_length'] for point in data['score_timeline']]
    fill_rates = [point['fill_rate'] for point in data['score_timeline']]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Score vs Temps
    ax1.plot(times, scores, 'b-', linewidth=2, marker='o', markersize=2)
    ax1.set_title('Score vs Temps')
    ax1.set_xlabel('Temps (s)')
    ax1.set_ylabel('Score')
    ax1.grid(True, alpha=0.3)
    
    # Longueur vs Temps
    ax2.plot(times, lengths, 'g-', linewidth=2, marker='s', markersize=2)
    ax2.set_title('Longueur du Serpent vs Temps')
    ax2.set_xlabel('Temps (s)')
    ax2.set_ylabel('Longueur')
    ax2.grid(True, alpha=0.3)
    
    # Taux de remplissage vs Temps
    ax3.plot(times, fill_rates, 'purple', linewidth=2, marker='^', markersize=2)
    ax3.set_title('Taux de Remplissage vs Temps')
    ax3.set_xlabel('Temps (s)')
    ax3.set_ylabel('Remplissage (%)')
    ax3.grid(True, alpha=0.3)
    
    # Score vs Longueur (corrélation)
    ax4.scatter(lengths, scores, c=times, cmap='viridis', s=30, alpha=0.7)
    ax4.set_title('Score vs Longueur (couleur = temps)')
    ax4.set_xlabel('Longueur du Serpent')
    ax4.set_ylabel('Score')
    ax4.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax4.scatter(lengths, scores, c=times, cmap='viridis', s=30, alpha=0.7), ax=ax4)
    cbar.set_label('Temps (s)')
    
    plt.suptitle(f'Analyse de Performance - {data["algorithm"]} (Score final: {data["final_score"]})', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()  # Fermer la figure au lieu de l'afficher

def calculate_statistics(data):
    """Calcule et affiche des statistiques sur la performance."""
    timeline = data['score_timeline']
    
    if len(timeline) < 2:
        print("Pas assez de données pour calculer les statistiques.")
        return
    
    times = [point['time'] for point in timeline]
    scores = [point['score'] for point in timeline]
    
    # Calcul du taux de score par seconde
    score_rates = []
    for i in range(1, len(timeline)):
        dt = times[i] - times[i-1]
        ds = scores[i] - scores[i-1]
        if dt > 0:
            score_rates.append(ds / dt)
    
    print("=" * 50)
    print("STATISTIQUES DE PERFORMANCE")
    print("=" * 50)
    print(f"Algorithme utilisé: {data['algorithm']}")
    print(f"Taille de la grille: {data['grid_size']}x{data['grid_size']}")
    print(f"Vitesse de jeu: {data['game_speed']} FPS")
    print(f"Résultat final: {data['game_result']}")
    print()
    print(f"Score final: {data['final_score']}")
    print(f"Temps total: {data['final_time']:.2f} secondes")
    print(f"Score moyen par seconde: {data['final_score'] / data['final_time']:.2f}")
    print()
    
    if score_rates:
        print(f"Taux de score max: {max(score_rates):.2f} points/s")
        print(f"Taux de score moyen: {np.mean(score_rates):.2f} points/s")
    
    final_fill_rate = timeline[-1]['fill_rate']
    print(f"Taux de remplissage final: {final_fill_rate:.1f}%")
    
    # Analyse de l'efficacité
    max_possible_score = data['grid_size'] * data['grid_size'] - 3  # -3 pour la taille initiale
    efficiency = (data['final_score'] / max_possible_score) * 100
    print(f"Efficacité de l'IA: {efficiency:.1f}% du score maximum possible")
    print("=" * 50)

def main():
    """Fonction principale pour générer tous les graphiques."""
    print("Analyseur de Performance Snake IA")
    print("=" * 40)
    
    # Charger les données
    data = load_game_data()
    if data is None:
        return
    
    print(f"Données chargées: {len(data['score_timeline'])} points de données")
    print(f"Jeu du: {datetime.fromtimestamp(data['start_time']).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Calculer et afficher les statistiques
    calculate_statistics(data)
    
    # Générer les graphiques
    print("\nGénération des graphiques...")
    
    try:
        plot_score_timeline(data)
        plot_snake_growth(data)
        plot_fill_rate(data)
        plot_performance_metrics(data)
        
        print("Tous les graphiques ont été générés et sauvegardés dans le dossier 'plots/'")
        
    except Exception as e:
        print(f"Erreur lors de la génération des graphiques: {e}")
        print("Assurez-vous que matplotlib est installé: pip install matplotlib")

if __name__ == "__main__":
    main()
