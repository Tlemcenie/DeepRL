"""
Environnement Grid World
Un monde en grille où l'agent peut se déplacer dans 4 directions
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Set
from .base_environment import BaseEnvironment


class GridWorld(BaseEnvironment):
    """
    Environnement Grid World : un monde en grille avec des obstacles et des objectifs.
    
    L'agent peut se déplacer dans 4 directions (haut, bas, gauche, droite).
    Certaines cases peuvent être des obstacles, des récompenses ou des pénalités.
    """
    
    def __init__(self, width: int = 5, height: int = 5, 
                 obstacles: Optional[List[Tuple[int, int]]] = None,
                 rewards: Optional[Dict[Tuple[int, int], float]] = None,
                 start_pos: Optional[Tuple[int, int]] = None,
                 terminal_states: Optional[List[Tuple[int, int]]] = None):
        """
        Args:
            width: Largeur de la grille
            height: Hauteur de la grille
            obstacles: Liste des positions (x, y) des obstacles
            rewards: Dictionnaire {(x, y): reward} des récompenses
            start_pos: Position de départ (x, y)
            terminal_states: Liste des états terminaux
        """
        super().__init__()
        
        self.width = width
        self.height = height
        
        # Configuration par défaut si non spécifié
        self.obstacles = set(obstacles) if obstacles else set()
        self.rewards = rewards if rewards else {
            (width-1, height-1): 1.0,  # Coin en bas à droite
            (0, height-1): -1.0        # Coin en bas à gauche
        }
        self.start_pos = start_pos if start_pos else (0, 0)
        self.terminal_states = set(terminal_states) if terminal_states else set(self.rewards.keys())
        
        # État actuel
        self.state = self.start_pos
        
        # Espaces d'action et d'observation
        self.action_space = [0, 1, 2, 3]  # 0:haut, 1:droite, 2:bas, 3:gauche
        self.action_names = {0: '↑', 1: '→', 2: '↓', 3: '←'}
        self.observation_space = [(x, y) for x in range(width) for y in range(height)
                                 if (x, y) not in self.obstacles]
        
    def reset(self) -> Tuple[int, int]:
        """Réinitialise l'environnement."""
        super().reset()
        self.state = self.start_pos
        self.history.append(self.state)
        return self.state
        
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict]:
        """
        Execute une action dans l'environnement.
        
        Args:
            action: 0=haut, 1=droite, 2=bas, 3=gauche
            
        Returns:
            tuple: (nouvel_état, récompense, terminé, info)
        """
        if self.done:
            raise ValueError("L'épisode est déjà terminé. Appelez reset().")
            
        if action not in self.action_space:
            raise ValueError(f"Action invalide {action}. Actions valides: {self.action_space}")
            
        # Sauvegarder l'état précédent
        old_state = self.state
        x, y = self.state
        
        # Calculer la nouvelle position
        if action == 0:  # Haut
            new_x, new_y = x, y - 1
        elif action == 1:  # Droite
            new_x, new_y = x + 1, y
        elif action == 2:  # Bas
            new_x, new_y = x, y + 1
        else:  # Gauche (action == 3)
            new_x, new_y = x - 1, y
            
        # Vérifier si la nouvelle position est valide
        if (0 <= new_x < self.width and 
            0 <= new_y < self.height and 
            (new_x, new_y) not in self.obstacles):
            self.state = (new_x, new_y)
        # Sinon, rester à la position actuelle
        
        # Calculer la récompense
        reward = self.rewards.get(self.state, -0.01)  # Petite pénalité par défaut
        
        # Vérifier si l'état est terminal
        self.done = self.state in self.terminal_states
        
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
            # Créer la grille
            grid = [[' . ' for _ in range(self.width)] for _ in range(self.height)]
            
            # Placer les obstacles
            for (x, y) in self.obstacles:
                grid[y][x] = ' # '
                
            # Placer les récompenses/pénalités
            for (x, y), reward in self.rewards.items():
                if reward > 0:
                    grid[y][x] = f'+{reward:.0f} '
                else:
                    grid[y][x] = f'{reward:.0f} '
                    
            # Placer l'agent
            x, y = self.state
            if self.state not in self.rewards:
                grid[y][x] = ' A '
            else:
                grid[y][x] = f'A{grid[y][x][1:]}'
                
            # Afficher
            output = f"\nGrid World ({self.width}x{self.height}):\n"
            output += "+" + "---+" * self.width + "\n"
            
            for row in grid:
                output += "|" + "|".join(row) + "|\n"
                output += "+" + "---+" * self.width + "\n"
                
            output += f"Position: {self.state}\n"
            output += f"Récompense totale: {self.episode_reward:.2f}\n"
            output += f"Étapes: {self.episode_steps}\n"
            output += f"Actions: 0=↑, 1=→, 2=↓, 3=←\n"
            
            if mode == 'human':
                print(output)
            return output
            
        elif mode == 'rgb_array':
            # Créer une image
            cell_size = 50
            img = np.ones((self.height * cell_size, self.width * cell_size, 3)) * 255
            
            # Dessiner la grille
            for y in range(self.height + 1):
                img[y * cell_size - 1:y * cell_size + 1, :, :] = 0
            for x in range(self.width + 1):
                img[:, x * cell_size - 1:x * cell_size + 1, :] = 0
                
            # Dessiner les éléments
            for y in range(self.height):
                for x in range(self.width):
                    cell_y = y * cell_size
                    cell_x = x * cell_size
                    
                    if (x, y) in self.obstacles:
                        # Obstacle en noir
                        img[cell_y+5:cell_y+45, cell_x+5:cell_x+45, :] = 0
                    elif (x, y) in self.rewards:
                        # Récompense/pénalité
                        if self.rewards[(x, y)] > 0:
                            # Récompense en vert
                            img[cell_y+10:cell_y+40, cell_x+10:cell_x+40, :] = [0, 255, 0]
                        else:
                            # Pénalité en rouge
                            img[cell_y+10:cell_y+40, cell_x+10:cell_x+40, :] = [255, 0, 0]
                            
                    if (x, y) == self.state:
                        # Agent en bleu
                        img[cell_y+15:cell_y+35, cell_x+15:cell_x+35, :] = [0, 0, 255]
                        
            return img.astype(np.uint8)
            
    def get_action_space(self) -> List[int]:
        """Retourne la liste des actions possibles."""
        return self.action_space
        
    def get_observation_space(self) -> List[Tuple[int, int]]:
        """Retourne l'espace des observations."""
        return self.observation_space
        
    def get_state_representation(self) -> str:
        """Retourne une représentation textuelle de l'état."""
        return str(self.state)
        
    def get_state_value_representation(self, values: Dict) -> str:
        """
        Affiche les valeurs d'état V(s) sur la grille.
        
        Args:
            values: Dictionnaire état -> valeur
        """
        output = "\nValeurs d'état V(s):\n"
        output += "+" + "-------+" * self.width + "\n"
        
        for y in range(self.height):
            row = "|"
            for x in range(self.width):
                if (x, y) in self.obstacles:
                    row += "  ###  |"
                elif (x, y) in values:
                    row += f"{values[(x, y)]:7.2f}|"
                else:
                    row += "   ?   |"
            output += row + "\n"
            output += "+" + "-------+" * self.width + "\n"
            
        return output
        
    def get_action_value_representation(self, q_values: Dict) -> str:
        """
        Affiche les Q-values Q(s,a) avec la meilleure action pour chaque état.
        
        Args:
            q_values: Dictionnaire (état, action) -> valeur
        """
        output = "\nMeilleures actions (basées sur Q-values):\n"
        output += "+" + "---+" * self.width + "\n"
        
        for y in range(self.height):
            row = "|"
            for x in range(self.width):
                if (x, y) in self.obstacles:
                    row += " # |"
                else:
                    # Trouver la meilleure action
                    best_action = None
                    best_value = float('-inf')
                    
                    for action in self.action_space:
                        if ((x, y), action) in q_values:
                            if q_values[((x, y), action)] > best_value:
                                best_value = q_values[((x, y), action)]
                                best_action = action
                                
                    if best_action is not None:
                        row += f" {self.action_names[best_action]} |"
                    else:
                        row += " ? |"
                        
            output += row + "\n"
            output += "+" + "---+" * self.width + "\n"
            
        # Ajouter les Q-values détaillées
        output += "\nQ-values détaillées:\n"
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) not in self.obstacles:
                    output += f"État ({x},{y}): "
                    for action in self.action_space:
                        if ((x, y), action) in q_values:
                            output += f"{self.action_names[action]}={q_values[((x, y), action)]:.2f} "
                    output += "\n"
                    
        return output
        
    def is_valid_position(self, x: int, y: int) -> bool:
        """Vérifie si une position est valide."""
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                (x, y) not in self.obstacles) 