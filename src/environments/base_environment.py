"""
Classe de base pour tous les environnements d'apprentissage par renforcement
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict, List, Optional
import numpy as np


class BaseEnvironment(ABC):
    """
    Classe abstraite définissant l'interface commune pour tous les environnements.
    Suit une interface similaire à OpenAI Gym pour la compatibilité.
    """
    
    def __init__(self):
        self.action_space = None
        self.observation_space = None
        self.state = None
        self.done = False
        self.info = {}
        self.episode_reward = 0
        self.episode_steps = 0
        self.history = []  # Pour sauvegarder l'historique des états
        
    @abstractmethod
    def reset(self) -> Any:
        """
        Réinitialise l'environnement et retourne l'état initial.
        
        Returns:
            observation: L'observation/état initial
        """
        self.done = False
        self.episode_reward = 0
        self.episode_steps = 0
        self.history = []
        
    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """
        Execute une action dans l'environnement.
        
        Args:
            action: L'action à exécuter
            
        Returns:
            observation: La nouvelle observation/état
            reward: La récompense obtenue
            done: Si l'épisode est terminé
            info: Dictionnaire d'informations supplémentaires
        """
        self.episode_steps += 1
        
    @abstractmethod
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Affiche l'état actuel de l'environnement.
        
        Args:
            mode: Mode de rendu ('human', 'rgb_array', 'ansi')
            
        Returns:
            Selon le mode, peut retourner une image ou None
        """
        pass
        
    @abstractmethod
    def get_action_space(self) -> List[Any]:
        """
        Retourne la liste des actions possibles.
        
        Returns:
            Liste des actions disponibles
        """
        pass
        
    @abstractmethod
    def get_observation_space(self) -> Any:
        """
        Retourne l'espace des observations.
        
        Returns:
            Description de l'espace des observations
        """
        pass
        
    def close(self):
        """
        Ferme l'environnement et libère les ressources.
        """
        pass
        
    def seed(self, seed: Optional[int] = None):
        """
        Définit la graine aléatoire pour la reproductibilité.
        
        Args:
            seed: Graine aléatoire
        """
        if seed is not None:
            np.random.seed(seed)
            
    def get_state_representation(self) -> str:
        """
        Retourne une représentation textuelle de l'état actuel.
        
        Returns:
            Représentation string de l'état
        """
        return str(self.state)
        
    def save_history(self, filepath: str):
        """
        Sauvegarde l'historique de l'épisode.
        
        Args:
            filepath: Chemin du fichier de sauvegarde
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.history, f)
            
    def load_history(self, filepath: str):
        """
        Charge un historique d'épisode.
        
        Args:
            filepath: Chemin du fichier à charger
        """
        import pickle
        with open(filepath, 'rb') as f:
            self.history = pickle.load(f)
            
    def play_manual(self):
        """
        Permet de jouer manuellement dans l'environnement.
        Mode interactif pour tester l'environnement.
        """
        print(f"Mode manuel pour {self.__class__.__name__}")
        print("Actions disponibles:", self.get_action_space())
        
        state = self.reset()
        self.render()
        
        while not self.done:
            print(f"\nÉtat actuel: {state}")
            print("Actions possibles:", self.get_action_space())
            
            action_input = input("Choisissez une action: ")
            try:
                # Essayer de convertir en int si possible
                if action_input.isdigit():
                    action = int(action_input)
                else:
                    action = action_input
                    
                state, reward, done, info = self.step(action)
                self.render()
                
                print(f"Récompense: {reward}")
                if done:
                    print(f"\nÉpisode terminé!")
                    print(f"Récompense totale: {self.episode_reward}")
                    
            except Exception as e:
                print(f"Erreur: {e}")
                print("Veuillez réessayer.")
                
    def get_state_value_representation(self, values: Dict) -> str:
        """
        Retourne une représentation de l'environnement avec les valeurs d'état.
        
        Args:
            values: Dictionnaire état -> valeur
            
        Returns:
            Représentation string avec les valeurs
        """
        return self.get_state_representation()
        
    def get_action_value_representation(self, q_values: Dict) -> str:
        """
        Retourne une représentation avec les Q-values.
        
        Args:
            q_values: Dictionnaire (état, action) -> valeur
            
        Returns:
            Représentation string avec les Q-values
        """
        return self.get_state_representation() 