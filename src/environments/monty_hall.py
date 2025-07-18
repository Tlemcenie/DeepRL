"""
Environnements Monty Hall
Le paradoxe de Monty Hall en version 3 portes (Level 1) et 5 portes (Level 2)
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from .base_environment import BaseEnvironment


class MontyHallBase(BaseEnvironment):
    """
    Classe de base pour les environnements Monty Hall.
    """
    
    def __init__(self, num_doors: int):
        super().__init__()
        
        self.num_doors = num_doors
        self.winning_door = None
        self.initial_choice = None
        self.doors_removed = []
        self.current_step = 0
        self.max_steps = num_doors - 1  # Nombre d'actions à prendre
        
        # État: (étape, choix_initial, portes_restantes)
        self.state = None
        
        # Espaces d'action dynamiques selon l'étape
        self.action_space = list(range(num_doors))  # Initial
        
    def reset(self) -> Tuple:
        """Réinitialise l'environnement."""
        super().reset()
        
        # Choisir aléatoirement la porte gagnante
        self.winning_door = np.random.randint(0, self.num_doors)
        self.initial_choice = None
        self.doors_removed = []
        self.current_step = 0
        
        # État initial: toutes les portes disponibles
        self.state = (0, None, tuple(range(self.num_doors)))
        self.history.append(self.state)
        
        return self.state
        
    def _remove_losing_door(self, available_doors: List[int]) -> int:
        """
        Retire une porte perdante parmi les portes disponibles.
        Ne retire jamais la porte gagnante ni le choix actuel de l'agent.
        """
        losing_doors = [d for d in available_doors 
                        if d != self.winning_door and d != self.initial_choice]
        
        if losing_doors:
            return np.random.choice(losing_doors)
        else:
            # Cas rare où l'agent a choisi la porte gagnante
            # On retire n'importe quelle autre porte
            other_doors = [d for d in available_doors if d != self.initial_choice]
            return np.random.choice(other_doors)
            
    def step(self, action: int) -> Tuple[Tuple, float, bool, Dict]:
        """
        Execute une action (choix de porte).
        
        Args:
            action: Numéro de la porte choisie
            
        Returns:
            tuple: (nouvel_état, récompense, terminé, info)
        """
        if self.done:
            raise ValueError("L'épisode est déjà terminé. Appelez reset().")
            
        step, initial_choice, remaining_doors = self.state
        remaining_doors = list(remaining_doors)
        
        if action not in remaining_doors:
            raise ValueError(f"Action invalide {action}. Portes disponibles: {remaining_doors}")
            
        reward = 0.0
        
        if step == 0:
            # Premier choix
            self.initial_choice = action
            
            # Retirer une porte perdante
            door_to_remove = self._remove_losing_door(remaining_doors)
            remaining_doors.remove(door_to_remove)
            self.doors_removed.append(door_to_remove)
            
            self.current_step += 1
            self.state = (self.current_step, self.initial_choice, tuple(remaining_doors))
            
        elif step < self.max_steps - 1:
            # Étapes intermédiaires (seulement pour Level 2)
            # Retirer une autre porte perdante
            door_to_remove = self._remove_losing_door(remaining_doors)
            remaining_doors.remove(door_to_remove)
            self.doors_removed.append(door_to_remove)
            
            self.current_step += 1
            self.state = (self.current_step, self.initial_choice, tuple(remaining_doors))
            
        else:
            # Dernier choix
            final_choice = action
            
            # Vérifier si l'agent a gagné
            if final_choice == self.winning_door:
                reward = 1.0
            else:
                reward = 0.0
                
            self.done = True
            self.state = (self.current_step + 1, self.initial_choice, (final_choice,))
            
        self.episode_reward += reward
        self.episode_steps += 1
        
        # Sauvegarder dans l'historique
        self.history.append({
            'state': (step, initial_choice, remaining_doors),
            'action': action,
            'reward': reward,
            'next_state': self.state,
            'done': self.done
        })
        
        info = {
            'step': self.current_step,
            'initial_choice': self.initial_choice,
            'doors_removed': self.doors_removed,
            'winning_door': self.winning_door if self.done else None,
            'episode_reward': self.episode_reward
        }
        
        return self.state, reward, self.done, info
        
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Affiche l'état actuel."""
        if mode == 'ansi' or mode == 'human':
            step, initial_choice, remaining_doors = self.state
            
            output = "\n" + "="*50 + "\n"
            output += f"Monty Hall - {self.num_doors} portes\n"
            output += "="*50 + "\n\n"
            
            # Afficher toutes les portes
            output += "Portes: "
            for i in range(self.num_doors):
                if i in self.doors_removed:
                    output += "[X] "  # Porte retirée
                elif i in remaining_doors:
                    if i == initial_choice:
                        output += f"[{i}*] "  # Choix initial
                    else:
                        output += f"[{i}] "  # Porte disponible
                else:
                    output += f"[{i}] "
            output += "\n\n"
            
            if not self.done:
                output += f"Étape {step + 1}/{self.max_steps}\n"
                if step == 0:
                    output += "Choisissez une porte initiale.\n"
                else:
                    output += f"Votre choix initial: Porte {initial_choice}\n"
                    output += f"Portes retirées: {self.doors_removed}\n"
                    output += "Voulez-vous garder votre choix ou changer?\n"
                output += f"Portes disponibles: {list(remaining_doors)}\n"
            else:
                output += "Partie terminée!\n"
                output += f"Porte gagnante: {self.winning_door}\n"
                output += f"Votre choix final: {remaining_doors[0]}\n"
                output += f"Résultat: {'GAGNÉ!' if self.episode_reward > 0 else 'PERDU!'}\n"
                
            if mode == 'human':
                print(output)
            return output
            
    def get_action_space(self) -> List[int]:
        """Retourne les actions possibles selon l'état actuel."""
        _, _, remaining_doors = self.state
        return list(remaining_doors)
        
    def get_observation_space(self) -> List:
        """Retourne l'espace des observations."""
        # Complexe à énumérer, on retourne une description
        return f"(step: 0-{self.max_steps}, initial_choice: 0-{self.num_doors-1}, remaining_doors: tuple)"


class MontyHallLevel1(MontyHallBase):
    """
    Monty Hall classique avec 3 portes.
    
    1. L'agent choisit une porte parmi 3
    2. Une porte perdante est retirée
    3. L'agent choisit de garder ou changer pour la porte restante
    """
    
    def __init__(self):
        super().__init__(num_doors=3)
        
    def get_optimal_strategy(self) -> str:
        """
        La stratégie optimale est de toujours changer de porte.
        Probabilité de gagner: 2/3 en changeant, 1/3 en gardant.
        """
        return "Toujours changer de porte au deuxième choix"


class MontyHallLevel2(MontyHallBase):
    """
    Monty Hall étendu avec 5 portes.
    
    1. L'agent choisit une porte parmi 5
    2-4. Une porte perdante est retirée à chaque étape
    5. L'agent fait son choix final entre les 2 portes restantes
    """
    
    def __init__(self):
        super().__init__(num_doors=5)
        
    def get_optimal_strategy(self) -> str:
        """
        La stratégie optimale est de toujours changer de porte au dernier choix.
        Probabilité de gagner: 4/5 en changeant, 1/5 en gardant.
        """
        return "Toujours changer de porte au dernier choix" 