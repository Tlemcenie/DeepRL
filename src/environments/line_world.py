"""
Environnement Line World
Un monde en ligne où l'agent peut se déplacer à gauche ou à droite
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from .base_environment import BaseEnvironment


class LineWorld(BaseEnvironment):
    """
    Environnement Line World : un monde linéaire avec des récompenses aux extrémités.
    
    L'agent commence au milieu et peut se déplacer à gauche (0) ou à droite (1).
    Récompenses : -1 à l'extrémité gauche, +1 à l'extrémité droite.
    """
    
    def __init__(self, size: int = 7):
        """
        Args:
            size: Nombre de positions dans la ligne (doit être impair)
        """
        super().__init__()
        
        if size % 2 == 0:
            size += 1  # S'assurer que la taille est impaire
            
        self.size = size
        self.start_position = size // 2  # Position centrale
        self.state = self.start_position
        
        # Définir les espaces d'action et d'observation
        self.action_space = [0, 1]  # 0: gauche, 1: droite
        self.observation_space = list(range(size))
        
        # Positions terminales
        self.terminal_states = [0, size - 1]
        
    def reset(self) -> int:
        """Réinitialise l'environnement."""
        super().reset()
        self.state = self.start_position
        self.history.append(self.state)
        return self.state
        
    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """
        Execute une action dans l'environnement.
        
        Args:
            action: 0 pour gauche, 1 pour droite
            
        Returns:
            tuple: (nouvel_état, récompense, terminé, info)
        """
        if self.done:
            raise ValueError("L'épisode est déjà terminé. Appelez reset().")
            
        if action not in self.action_space:
            raise ValueError(f"Action invalide {action}. Actions valides: {self.action_space}")
            
        # Sauvegarder l'état précédent
        old_state = self.state
        
        # Effectuer l'action
        if action == 0 and self.state > 0:  # Aller à gauche
            self.state -= 1
        elif action == 1 and self.state < self.size - 1:  # Aller à droite
            self.state += 1
            
        # Calculer la récompense
        reward = 0.0
        if self.state == 0:  # Extrémité gauche
            reward = -1.0
            self.done = True
        elif self.state == self.size - 1:  # Extrémité droite
            reward = 1.0
            self.done = True
            
        self.episode_reward += reward
        self.episode_steps += 1
        
        # Sauvegarder dans l'historique
        self.history.append({
            'state': old_state,
            'action': action,
            'reward': reward,
            'next_state': self.state,
            'done': self.done
        })
        
        info = {
            'position': self.state,
            'episode_reward': self.episode_reward,
            'episode_steps': self.episode_steps
        }
        
        return self.state, reward, self.done, info
        
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Affiche l'état actuel de l'environnement.
        
        Args:
            mode: Mode de rendu ('human', 'ansi', 'rgb_array')
        """
        if mode == 'ansi' or mode == 'human':
            # Créer la représentation visuelle
            line = ['.'] * self.size
            line[self.state] = 'A'  # A pour Agent
            line[0] = 'L'  # L pour Left (récompense négative)
            line[-1] = 'R'  # R pour Right (récompense positive)
            
            # Afficher
            display = ' '.join(line)
            values = ' '.join([f"{i:2d}" for i in range(self.size)])
            
            output = f"\nLine World (taille={self.size}):\n"
            output += f"Positions: {values}\n"
            output += f"État:      {display}\n"
            output += f"Position actuelle: {self.state}\n"
            output += f"Récompense totale: {self.episode_reward:.1f}\n"
            output += f"Étapes: {self.episode_steps}\n"
            
            if mode == 'human':
                print(output)
            return output
            
        elif mode == 'rgb_array':
            # Créer une image simple
            img_height = 100
            img_width = 50 * self.size
            img = np.ones((img_height, img_width, 3)) * 255
            
            # Dessiner les positions
            for i in range(self.size):
                x = i * 50
                # Positions terminales en couleur
                if i == 0:
                    img[20:80, x:x+40, :] = [255, 0, 0]  # Rouge
                elif i == self.size - 1:
                    img[20:80, x:x+40, :] = [0, 255, 0]  # Vert
                else:
                    img[20:80, x:x+40, :] = [200, 200, 200]  # Gris
                    
                # Position de l'agent
                if i == self.state:
                    img[30:70, x+5:x+35, :] = [0, 0, 255]  # Bleu
                    
            return img.astype(np.uint8)
            
    def get_action_space(self) -> List[int]:
        """Retourne la liste des actions possibles."""
        return self.action_space
        
    def get_observation_space(self) -> List[int]:
        """Retourne l'espace des observations."""
        return self.observation_space
        
    def get_state_representation(self) -> str:
        """Retourne une représentation textuelle de l'état."""
        return str(self.state)
        
    def get_state_value_representation(self, values: Dict) -> str:
        """
        Affiche les valeurs d'état V(s).
        
        Args:
            values: Dictionnaire état -> valeur
        """
        output = "\nValeurs d'état V(s):\n"
        line_v = []
        for i in range(self.size):
            if i in values:
                line_v.append(f"{values[i]:5.2f}")
            else:
                line_v.append("  ?  ")
                
        output += ' '.join(line_v) + "\n"
        output += ' '.join([f"  {i:2d} " for i in range(self.size)]) + "\n"
        
        return output
        
    def get_action_value_representation(self, q_values: Dict) -> str:
        """
        Affiche les Q-values Q(s,a).
        
        Args:
            q_values: Dictionnaire (état, action) -> valeur
        """
        output = "\nQ-values Q(s,a):\n"
        output += "État | Gauche(0) | Droite(1)\n"
        output += "-" * 30 + "\n"
        
        for state in range(self.size):
            left_q = q_values.get((state, 0), "?")
            right_q = q_values.get((state, 1), "?")
            
            if isinstance(left_q, float):
                left_q = f"{left_q:7.3f}"
            if isinstance(right_q, float):
                right_q = f"{right_q:7.3f}"
                
            output += f" {state:2d}  | {left_q:>9} | {right_q:>9}\n"
            
        return output
        
    def get_optimal_policy(self) -> Dict[int, int]:
        """
        Retourne la politique optimale pour Line World.
        
        Returns:
            Dictionnaire état -> action optimale
        """
        policy = {}
        for state in range(self.size):
            if state < self.start_position:
                policy[state] = 0  # Aller à gauche pour atteindre -1
            else:
                policy[state] = 1  # Aller à droite pour atteindre +1
        return policy 