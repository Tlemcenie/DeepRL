"""
Classe de base pour tous les algorithmes d'apprentissage par renforcement
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import time


class BaseAlgorithm(ABC):
    """
    Classe abstraite définissant l'interface commune pour tous les algorithmes RL.
    """
    
    def __init__(self, env, **kwargs):
        """
        Args:
            env: Environnement d'apprentissage
            **kwargs: Hyperparamètres spécifiques à l'algorithme
        """
        self.env = env
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'steps': [],
            'values': [],
            'timestamps': []
        }
        self.policy = {}
        self.value_function = {}
        self.q_function = {}
        self.trained = False
        
        # Hyperparamètres communs
        self.gamma = kwargs.get('gamma', 0.99)  # Facteur de discount
        self.episodes = kwargs.get('episodes', 1000)  # Nombre d'épisodes
        self.seed = kwargs.get('seed', None)
        
        if self.seed is not None:
            np.random.seed(self.seed)
            self.env.seed(self.seed)
            
    @abstractmethod
    def train(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Entraîne l'algorithme sur l'environnement.
        
        Args:
            verbose: Afficher la progression
            
        Returns:
            Dictionnaire contenant les résultats d'entraînement
        """
        self.trained = True
        
    @abstractmethod
    def get_policy(self) -> Dict:
        """
        Retourne la politique apprise.
        
        Returns:
            Dictionnaire état -> action
        """
        pass
        
    @abstractmethod
    def get_value_function(self) -> Dict:
        """
        Retourne la fonction de valeur V(s).
        
        Returns:
            Dictionnaire état -> valeur
        """
        pass
        
    @abstractmethod
    def get_q_function(self) -> Dict:
        """
        Retourne la fonction Q(s,a).
        
        Returns:
            Dictionnaire (état, action) -> valeur
        """
        pass
        
    def act(self, state: Any, explore: bool = False) -> Any:
        """
        Choisit une action selon la politique.
        
        Args:
            state: État actuel
            explore: Si True, peut explorer (epsilon-greedy)
            
        Returns:
            Action à effectuer
        """
        if not self.trained and not explore:
            raise ValueError("L'algorithme doit être entraîné avant d'agir.")
            
        if state in self.policy:
            return self.policy[state]
        else:
            # Action aléatoire si état non vu
            return np.random.choice(self.env.get_action_space())
            
    def evaluate_policy(self, num_episodes: int = 100, max_steps: int = 1000) -> Dict[str, float]:
        """
        Évalue la politique actuelle.
        
        Args:
            num_episodes: Nombre d'épisodes d'évaluation
            max_steps: Nombre max de pas par épisode
            
        Returns:
            Statistiques de performance
        """
        total_rewards = []
        episode_lengths = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            steps = 0
            
            for _ in range(max_steps):
                action = self.act(state)
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                steps += 1
                
                if done:
                    break
                    
            total_rewards.append(episode_reward)
            episode_lengths.append(steps)
            
        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'min_reward': np.min(total_rewards),
            'max_reward': np.max(total_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'success_rate': np.mean([r > 0 for r in total_rewards])
        }
        
    def save(self, filepath: str):
        """
        Sauvegarde l'algorithme entraîné.
        
        Args:
            filepath: Chemin du fichier de sauvegarde
        """
        data = {
            'algorithm': self.__class__.__name__,
            'policy': self.policy,
            'value_function': self.value_function,
            'q_function': self.q_function,
            'training_history': self.training_history,
            'hyperparameters': {
                'gamma': self.gamma,
                'episodes': self.episodes,
                'seed': self.seed
            },
            'trained': self.trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
    def load(self, filepath: str):
        """
        Charge un algorithme sauvegardé.
        
        Args:
            filepath: Chemin du fichier à charger
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        if data['algorithm'] != self.__class__.__name__:
            raise ValueError(f"Incompatibilité: tentative de charger {data['algorithm']} "
                           f"dans {self.__class__.__name__}")
            
        self.policy = data['policy']
        self.value_function = data['value_function']
        self.q_function = data['q_function']
        self.training_history = data['training_history']
        self.gamma = data['hyperparameters']['gamma']
        self.episodes = data['hyperparameters']['episodes']
        self.seed = data['hyperparameters']['seed']
        self.trained = data['trained']
        
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Affiche l'historique d'entraînement.
        
        Args:
            save_path: Si fourni, sauvegarde le graphique
        """
        if not self.training_history['episodes']:
            print("Aucun historique d'entraînement disponible.")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Récompenses par épisode
        ax1.plot(self.training_history['episodes'], 
                self.training_history['rewards'])
        ax1.set_xlabel('Épisode')
        ax1.set_ylabel('Récompense totale')
        ax1.set_title('Récompense par épisode')
        ax1.grid(True)
        
        # Moyenne mobile
        window = min(100, len(self.training_history['rewards']) // 10)
        if window > 1:
            moving_avg = np.convolve(self.training_history['rewards'], 
                                    np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(self.training_history['rewards'])), 
                    moving_avg, 'r-', linewidth=2, label=f'Moyenne mobile ({window} épisodes)')
            ax1.legend()
        
        # Nombre de pas par épisode
        ax2.plot(self.training_history['episodes'], 
                self.training_history['steps'])
        ax2.set_xlabel('Épisode')
        ax2.set_ylabel('Nombre de pas')
        ax2.set_title('Durée des épisodes')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
    def demonstrate_policy(self, num_steps: int = 20, delay: float = 0.5):
        """
        Démontre la politique apprise pas à pas.
        
        Args:
            num_steps: Nombre max de pas
            delay: Délai entre les pas (secondes)
        """
        if not self.trained:
            raise ValueError("L'algorithme doit être entraîné avant la démonstration.")
            
        print("\n" + "="*50)
        print("DÉMONSTRATION DE LA POLITIQUE APPRISE")
        print("="*50 + "\n")
        
        state = self.env.reset()
        self.env.render()
        
        for step in range(num_steps):
            print(f"\nÉtape {step + 1}:")
            print(f"État actuel: {state}")
            
            action = self.act(state)
            print(f"Action choisie: {action}")
            
            state, reward, done, info = self.env.step(action)
            
            print(f"Récompense: {reward}")
            self.env.render()
            
            if done:
                print("\n" + "="*50)
                print("ÉPISODE TERMINÉ!")
                print(f"Récompense totale: {info.get('episode_reward', 'N/A')}")
                print(f"Nombre de pas: {info.get('episode_steps', 'N/A')}")
                print("="*50)
                break
                
            time.sleep(delay)
            
    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Retourne les hyperparamètres de l'algorithme.
        
        Returns:
            Dictionnaire des hyperparamètres
        """
        return {
            'gamma': self.gamma,
            'episodes': self.episodes,
            'seed': self.seed
        } 