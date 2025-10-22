#!/usr/bin/env python3
"""
Script de comparaison entre l'IA (Deep Q-Learning) et l'algorithme A*.
Compare les performances sur plusieurs parties pour avoir des statistiques fiables.
"""

import pygame
import time
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import sys
import importlib.util

# Charger les modules depuis les fichiers avec tirets
def load_module_from_file(module_name, file_path):
    """Charge un module Python depuis un fichier avec un nom non-standard."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Charger les modules
snake_ia = load_module_from_file('snake_ia', 'snake-ia.py')
snake_algo = load_module_from_file('snake_algo', 'snake-algo.py')

# Importer les classes nécessaires
SnakeGameAI = snake_ia.SnakeGameAI
Agent = snake_ia.Agent
SnakeGameAlgo = snake_algo.SnakeGameAlgo
PathFinder = snake_algo.PathFinder
Algorithm = snake_algo.Algorithm

# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_GAMES = 50  # Nombre de parties pour chaque méthode
SPEED_DELAY = 0  # Pas de délai pour aller plus vite (mettre 50 pour voir visuellement)

# ============================================================================
# CLASSE DE COMPARAISON
# ============================================================================

class SnakeComparison:
    """Compare l'IA et l'algorithme A* sur plusieurs parties."""
    
    def __init__(self, num_games=NUM_GAMES, visual=False):
        self.num_games = num_games
        self.visual = visual
        self.results = {
            'ia': {'scores': [], 'steps': [], 'times': []},
            'astar': {'scores': [], 'steps': [], 'times': []}
        }
    
    def run_ia_games(self):
        """Lance plusieurs parties avec l'IA Deep Q-Learning."""
        print("\n" + "="*60)
        print("🤖 TEST DE L'IA (Deep Q-Learning)")
        print("="*60)
        
        # Initialiser l'agent (charge le modèle si disponible)
        agent = Agent(load_model=True)
        
        for game_num in range(self.num_games):
            game = SnakeGameAI(headless=not self.visual)
            start_time = time.time()
            steps = 0
            
            while True:
                # Obtenir l'état actuel
                state_old = agent.get_state(game)
                
                # Obtenir l'action (exploitation pure, pas d'exploration)
                final_move = agent.get_action_greedy(state_old)
                
                # Effectuer l'action
                reward, done, score = game.play_step(final_move)
                steps += 1
                
                if self.visual:
                    pygame.time.delay(SPEED_DELAY)
                
                if done:
                    elapsed = time.time() - start_time
                    self.results['ia']['scores'].append(score)
                    self.results['ia']['steps'].append(steps)
                    self.results['ia']['times'].append(elapsed)
                    
                    print(f"Partie {game_num+1}/{self.num_games} - Score: {score:3d} - Steps: {steps:4d} - Temps: {elapsed:.2f}s")
                    break
            
            if self.visual:
                pygame.quit()
        
        print(f"\n✅ IA terminée : {self.num_games} parties")
    
    def run_astar_games(self):
        """Lance plusieurs parties avec l'algorithme A*."""
        print("\n" + "="*60)
        print("� TEST DE L'ALGORITHME A*")
        print("="*60)
        
        for game_num in range(self.num_games):
            # Créer le jeu avec A* intégré
            game = SnakeGameAlgo(algorithm=Algorithm.ASTAR, visualize_path=False)
            start_time = time.time()
            steps = 0
            
            # Boucle de jeu
            running = True
            while running and not game.game_over:
                # Gérer les événements pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                
                # Exécuter un pas (A* calcule automatiquement le chemin)
                game.step()
                steps += 1
                
                # Affichage conditionnel
                if self.visual:
                    game.draw()
                    pygame.time.delay(SPEED_DELAY)
                    game.clock.tick(40)
                else:
                    # Mode rapide sans affichage
                    if steps % 50 == 0:  # Afficher tous les 50 steps pour éviter le freeze
                        game.draw()
                    game.clock.tick(1000)  # Vitesse maximale
            
            # Récupérer les résultats
            elapsed = time.time() - start_time
            score = game.snake.score
            
            self.results['astar']['scores'].append(score)
            self.results['astar']['steps'].append(steps)
            self.results['astar']['times'].append(elapsed)
            
            print(f"Partie {game_num+1}/{self.num_games} - Score: {score:3d} - Steps: {steps:4d} - Temps: {elapsed:.2f}s")
            
            pygame.quit()
        
        print(f"\n✅ A* terminé : {self.num_games} parties")
    
    def print_statistics(self):
        """Affiche les statistiques de comparaison."""
        print("\n" + "="*60)
        print("📊 RÉSULTATS DE LA COMPARAISON")
        print("="*60)
        
        for method_name, method_key in [("IA (Deep Q-Learning)", "ia"), ("Algorithme A*", "astar")]:
            scores = self.results[method_key]['scores']
            steps = self.results[method_key]['steps']
            times = self.results[method_key]['times']
            
            if not scores:
                print(f"\n❌ {method_name}: Aucune donnée")
                continue
            
            print(f"\n🔹 {method_name}")
            print(f"   Nombre de parties: {len(scores)}")
            print(f"\n   📈 SCORES:")
            print(f"      Minimum:  {min(scores):3d}")
            print(f"      Maximum:  {max(scores):3d}")
            print(f"      Moyenne:  {np.mean(scores):6.2f}")
            print(f"      Médiane:  {np.median(scores):6.2f}")
            print(f"      Écart-type: {np.std(scores):6.2f}")
            
            print(f"\n   👣 STEPS (nombre de mouvements):")
            print(f"      Minimum:  {min(steps):4d}")
            print(f"      Maximum:  {max(steps):4d}")
            print(f"      Moyenne:  {np.mean(steps):8.2f}")
            
            print(f"\n   ⏱️  TEMPS (secondes):")
            print(f"      Total:    {sum(times):8.2f}s")
            print(f"      Moyenne:  {np.mean(times):8.2f}s")
        
        # Comparaison directe
        print("\n" + "="*60)
        print("🏆 COMPARAISON DIRECTE")
        print("="*60)
        
        ia_scores = self.results['ia']['scores']
        astar_scores = self.results['astar']['scores']
        
        if ia_scores and astar_scores:
            ia_avg = np.mean(ia_scores)
            astar_avg = np.mean(astar_scores)
            
            print(f"\n📊 Score moyen:")
            print(f"   IA:    {ia_avg:6.2f}")
            print(f"   A*:    {astar_avg:6.2f}")
            print(f"   Différence: {abs(ia_avg - astar_avg):6.2f} ({((ia_avg - astar_avg) / astar_avg * 100):+.1f}%)")
            
            if ia_avg > astar_avg:
                print(f"\n🏆 L'IA gagne avec {ia_avg - astar_avg:.2f} points de plus en moyenne !")
            elif astar_avg > ia_avg:
                print(f"\n🏆 A* gagne avec {astar_avg - ia_avg:.2f} points de plus en moyenne !")
            else:
                print(f"\n🤝 Égalité parfaite !")
            
            # Taux de victoires (score > seuil)
            threshold = 10
            ia_wins = sum(1 for s in ia_scores if s >= threshold)
            astar_wins = sum(1 for s in astar_scores if s >= threshold)
            
            print(f"\n🎯 Parties avec score ≥ {threshold}:")
            print(f"   IA:    {ia_wins}/{len(ia_scores)} ({ia_wins/len(ia_scores)*100:.1f}%)")
            print(f"   A*:    {astar_wins}/{len(astar_scores)} ({astar_wins/len(astar_scores)*100:.1f}%)")
    
    def plot_results(self):
        """Génère des graphiques de comparaison."""
        print("\n📊 Génération des graphiques...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Comparaison IA vs A*', fontsize=16, fontweight='bold')
        
        ia_scores = self.results['ia']['scores']
        astar_scores = self.results['astar']['scores']
        
        # Graphique 1: Évolution des scores
        ax1 = axes[0, 0]
        if ia_scores:
            ax1.plot(range(1, len(ia_scores)+1), ia_scores, 'b-', label='IA', alpha=0.7)
        if astar_scores:
            ax1.plot(range(1, len(astar_scores)+1), astar_scores, 'r-', label='A*', alpha=0.7)
        ax1.set_xlabel('Numéro de partie')
        ax1.set_ylabel('Score')
        ax1.set_title('Évolution des scores au fil des parties')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Graphique 2: Distribution des scores (histogramme)
        ax2 = axes[0, 1]
        if ia_scores and astar_scores:
            bins = range(0, max(max(ia_scores), max(astar_scores)) + 2)
            ax2.hist(ia_scores, bins=bins, alpha=0.5, label='IA', color='blue', edgecolor='black')
            ax2.hist(astar_scores, bins=bins, alpha=0.5, label='A*', color='red', edgecolor='black')
        elif ia_scores:
            bins = range(0, max(ia_scores) + 2)
            ax2.hist(ia_scores, bins=bins, alpha=0.7, label='IA', color='blue', edgecolor='black')
        elif astar_scores:
            bins = range(0, max(astar_scores) + 2)
            ax2.hist(astar_scores, bins=bins, alpha=0.7, label='A*', color='red', edgecolor='black')
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Fréquence')
        ax2.set_title('Distribution des scores')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Graphique 3: Box plot comparatif
        ax3 = axes[1, 0]
        data_to_plot = []
        labels = []
        if ia_scores:
            data_to_plot.append(ia_scores)
            labels.append('IA')
        if astar_scores:
            data_to_plot.append(astar_scores)
            labels.append('A*')
        
        if data_to_plot:
            bp = ax3.boxplot(data_to_plot, labels=labels, patch_artist=True)
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
        ax3.set_ylabel('Score')
        ax3.set_title('Comparaison statistique des scores')
        ax3.grid(True, alpha=0.3)
        
        # Graphique 4: Moyennes mobiles
        ax4 = axes[1, 1]
        window = 5  # Fenêtre de moyenne mobile
        if ia_scores and len(ia_scores) >= window:
            ia_ma = np.convolve(ia_scores, np.ones(window)/window, mode='valid')
            ax4.plot(range(window, len(ia_scores)+1), ia_ma, 'b-', label='IA (moyenne mobile)', linewidth=2)
        if astar_scores and len(astar_scores) >= window:
            astar_ma = np.convolve(astar_scores, np.ones(window)/window, mode='valid')
            ax4.plot(range(window, len(astar_scores)+1), astar_ma, 'r-', label='A* (moyenne mobile)', linewidth=2)
        ax4.set_xlabel('Numéro de partie')
        ax4.set_ylabel('Score moyen')
        ax4.set_title(f'Tendance (moyenne mobile sur {window} parties)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Sauvegarder le graphique
        filename = 'comparison_ia_vs_astar.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✅ Graphiques sauvegardés dans '{filename}'")
        
        # Afficher le graphique
        plt.show()
    
    def run_full_comparison(self):
        """Lance la comparaison complète."""
        print("\n" + "="*60)
        print("🎮 COMPARAISON IA vs A*")
        print("="*60)
        print(f"Nombre de parties par méthode: {self.num_games}")
        print(f"Mode visuel: {'Activé' if self.visual else 'Désactivé'}")
        
        # Tester l'IA
        self.run_ia_games()
        
        # Tester A*
        self.run_astar_games()
        
        # Afficher les statistiques
        self.print_statistics()
        
        # Générer les graphiques
        self.plot_results()

# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """Point d'entrée du programme."""
    # Parser les arguments
    visual = '--visual' in sys.argv
    num_games = NUM_GAMES
    
    # Chercher le nombre de parties dans les arguments
    for arg in sys.argv:
        if arg.startswith('--games='):
            try:
                num_games = int(arg.split('=')[1])
            except ValueError:
                print(f"⚠️  Nombre de parties invalide: {arg}")
    
    print("\n🐍 SNAKE - COMPARAISON IA vs A*")
    print("Appuyez sur Ctrl+C pour arrêter à tout moment\n")
    
    try:
        comparison = SnakeComparison(num_games=num_games, visual=visual)
        comparison.run_full_comparison()
        
        print("\n✅ Comparaison terminée avec succès !")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Comparaison interrompue par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
