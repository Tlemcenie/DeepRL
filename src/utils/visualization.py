"""
Utilitaires de visualisation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import pandas as pd
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches


class Visualizer:
    """
    Classe pour créer des visualisations avancées.
    """
    
    def __init__(self):
        # Style par défaut
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def plot_learning_curves(self, 
                           results_list: List[Dict[str, Any]], 
                           title: str = "Courbes d'apprentissage",
                           save_path: Optional[str] = None):
        """
        Trace les courbes d'apprentissage pour plusieurs algorithmes.
        
        Args:
            results_list: Liste des résultats d'expérimentation
            title: Titre du graphique
            save_path: Chemin pour sauvegarder
        """
        plt.figure(figsize=(12, 8))
        
        for result in results_list:
            if 'training_history' in result and 'rewards' in result['training_history']:
                rewards = result['training_history']['rewards']
                episodes = result['training_history']['episodes']
                label = f"{result['algorithm']} - {result['environment']}"
                
                # Courbe principale
                plt.plot(episodes, rewards, alpha=0.3, linewidth=0.5)
                
                # Moyenne mobile
                window = min(100, len(rewards) // 10)
                if window > 1:
                    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    plt.plot(episodes[window-1:], moving_avg, label=label, linewidth=2)
                    
        plt.xlabel('Épisode')
        plt.ylabel('Récompense totale')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
    def plot_performance_comparison(self,
                                  results_df: pd.DataFrame,
                                  metric: str = 'Mean Reward',
                                  save_path: Optional[str] = None):
        """
        Compare les performances des algorithmes.
        
        Args:
            results_df: DataFrame avec les résultats
            metric: Métrique à comparer
            save_path: Chemin pour sauvegarder
        """
        plt.figure(figsize=(14, 8))
        
        # Préparer les données
        pivot_df = results_df.pivot(index='Environment', columns='Algorithm', values=metric)
        
        # Barplot groupé
        ax = pivot_df.plot(kind='bar', width=0.8)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel(metric)
        plt.title(f'Comparaison de {metric} par Algorithme et Environnement')
        plt.legend(title='Algorithme', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Ajouter les valeurs sur les barres
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3)
            
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
    def plot_hyperparameter_analysis(self,
                                   param_analysis: Dict[str, Any],
                                   save_path: Optional[str] = None):
        """
        Visualise l'analyse des hyperparamètres.
        
        Args:
            param_analysis: Résultats de l'analyse des paramètres
            save_path: Chemin pour sauvegarder
        """
        n_params = len(param_analysis)
        fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 5))
        
        if n_params == 1:
            axes = [axes]
            
        for idx, (param_name, data) in enumerate(param_analysis.items()):
            ax = axes[idx]
            
            # Scatter plot
            ax.scatter(data['values'], data['scores'], alpha=0.6)
            
            # Ligne de tendance si possible
            if len(set(data['values'])) > 1:
                z = np.polyfit(data['values'], data['scores'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(data['values']), max(data['values']), 100)
                ax.plot(x_line, p(x_line), "r--", alpha=0.8)
                
            ax.set_xlabel(param_name)
            ax.set_ylabel('Score')
            ax.set_title(f'{param_name}\nCorr: {data["correlation"]:.3f}')
            ax.grid(True)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
    def visualize_value_function_heatmap(self,
                                       value_function: Dict,
                                       env_type: str = 'grid',
                                       save_path: Optional[str] = None):
        """
        Visualise la fonction de valeur comme heatmap.
        
        Args:
            value_function: Dictionnaire état -> valeur
            env_type: Type d'environnement ('grid' ou 'line')
            save_path: Chemin pour sauvegarder
        """
        if env_type == 'line':
            # Pour Line World
            states = sorted([s for s in value_function.keys() if isinstance(s, int)])
            values = [value_function[s] for s in states]
            
            plt.figure(figsize=(10, 4))
            plt.bar(states, values, color=plt.cm.RdYlGn([(v-min(values))/(max(values)-min(values)) for v in values]))
            plt.xlabel('Position')
            plt.ylabel('Valeur V(s)')
            plt.title('Fonction de Valeur - Line World')
            
            # Ajouter les valeurs sur les barres
            for i, v in enumerate(values):
                plt.text(states[i], v, f'{v:.2f}', ha='center', va='bottom')
                
        elif env_type == 'grid':
            # Pour Grid World
            # Extraire les dimensions
            max_x = max(s[0] for s in value_function.keys() if isinstance(s, tuple))
            max_y = max(s[1] for s in value_function.keys() if isinstance(s, tuple))
            
            # Créer la matrice
            grid = np.zeros((max_y + 1, max_x + 1))
            for (x, y), value in value_function.items():
                if isinstance((x, y), tuple):
                    grid[y, x] = value
                    
            plt.figure(figsize=(8, 8))
            sns.heatmap(grid, annot=True, fmt='.2f', cmap='RdYlGn', center=0)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Fonction de Valeur - Grid World')
            
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
    def visualize_policy(self,
                       policy: Dict,
                       env_type: str = 'grid',
                       action_names: Optional[Dict] = None,
                       save_path: Optional[str] = None):
        """
        Visualise la politique.
        
        Args:
            policy: Dictionnaire état -> action
            env_type: Type d'environnement
            action_names: Dictionnaire action -> nom
            save_path: Chemin pour sauvegarder
        """
        if action_names is None:
            action_names = {0: '↑', 1: '→', 2: '↓', 3: '←'}
            
        if env_type == 'grid':
            # Pour Grid World
            max_x = max(s[0] for s in policy.keys() if isinstance(s, tuple))
            max_y = max(s[1] for s in policy.keys() if isinstance(s, tuple))
            
            fig, ax = plt.subplots(figsize=(max_x+2, max_y+2))
            
            # Grille
            for x in range(max_x + 2):
                ax.axvline(x, color='black', linewidth=0.5)
            for y in range(max_y + 2):
                ax.axhline(y, color='black', linewidth=0.5)
                
            # Actions
            for (x, y), action in policy.items():
                if isinstance((x, y), tuple):
                    if action in action_names:
                        ax.text(x + 0.5, max_y - y + 0.5, action_names[action], 
                               ha='center', va='center', fontsize=20)
                    else:
                        ax.text(x + 0.5, max_y - y + 0.5, str(action), 
                               ha='center', va='center', fontsize=12)
                        
            ax.set_xlim(0, max_x + 1)
            ax.set_ylim(0, max_y + 1)
            ax.set_aspect('equal')
            ax.set_title('Politique Optimale')
            ax.axis('off')
            
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
    def create_training_animation(self,
                                training_history: Dict,
                                env_name: str,
                                save_path: Optional[str] = None):
        """
        Crée une animation de l'entraînement.
        
        Args:
            training_history: Historique d'entraînement
            env_name: Nom de l'environnement
            save_path: Chemin pour sauvegarder
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        episodes = training_history['episodes']
        rewards = training_history['rewards']
        steps = training_history['steps']
        
        # Lignes vides pour l'animation
        line1, = ax1.plot([], [], 'b-', linewidth=2)
        line2, = ax2.plot([], [], 'r-', linewidth=2)
        
        ax1.set_xlim(0, max(episodes))
        ax1.set_ylim(min(rewards) - 0.1, max(rewards) + 0.1)
        ax1.set_xlabel('Épisode')
        ax1.set_ylabel('Récompense')
        ax1.set_title(f'Entraînement sur {env_name}')
        ax1.grid(True)
        
        ax2.set_xlim(0, max(episodes))
        ax2.set_ylim(0, max(steps) + 1)
        ax2.set_xlabel('Épisode')
        ax2.set_ylabel('Nombre de pas')
        ax2.grid(True)
        
        def animate(frame):
            line1.set_data(episodes[:frame], rewards[:frame])
            line2.set_data(episodes[:frame], steps[:frame])
            return line1, line2
            
        anim = FuncAnimation(fig, animate, frames=len(episodes), 
                           interval=50, blit=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=20)
        else:
            plt.show()
            
        return anim 