"""
Environnement Two Round Rock Paper Scissors
L'adversaire joue aléatoirement au round 1, puis joue le choix de l'agent au round 2
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from .base_environment import BaseEnvironment


class TwoRoundRockPaperScissors(BaseEnvironment):
    """
    Jeu de Pierre-Feuille-Ciseaux en 2 rounds.
    
    Round 1: L'adversaire joue aléatoirement
    Round 2: L'adversaire joue FORCÉMENT le choix de l'agent au round 1
    
    Actions: 0=Pierre, 1=Feuille, 2=Ciseaux
    Récompenses: +1 (victoire), -1 (défaite), 0 (égalité)
    """
    
    def __init__(self):
        super().__init__()
        
        # Espaces d'action
        self.action_space = [0, 1, 2]  # Pierre, Feuille, Ciseaux
        self.action_names = {0: 'Pierre', 1: 'Feuille', 2: 'Ciseaux'}
        self.action_symbols = {0: '✊', 1: '✋', 2: '✌'}
        
        # État: (round, choix_agent_round1)
        # round: 0 ou 1
        # choix_agent_round1: None, 0, 1 ou 2
        self.state = (0, None)
        self.round = 0
        self.agent_choice_round1 = None
        self.opponent_choice_round1 = None
        self.opponent_choice_round2 = None
        
        # Historique des rounds
        self.round_history = []
        
        # Matrice des résultats (agent vs adversaire)
        # 0: égalité, 1: agent gagne, -1: agent perd
        self.outcome_matrix = {
            (0, 0): 0,   # Pierre vs Pierre
            (0, 1): -1,  # Pierre vs Feuille
            (0, 2): 1,   # Pierre vs Ciseaux
            (1, 0): 1,   # Feuille vs Pierre
            (1, 1): 0,   # Feuille vs Feuille
            (1, 2): -1,  # Feuille vs Ciseaux
            (2, 0): -1,  # Ciseaux vs Pierre
            (2, 1): 1,   # Ciseaux vs Feuille
            (2, 2): 0    # Ciseaux vs Ciseaux
        }
        
    def reset(self) -> Tuple[int, Optional[int]]:
        """Réinitialise l'environnement."""
        super().reset()
        self.round = 0
        self.agent_choice_round1 = None
        self.opponent_choice_round1 = None
        self.opponent_choice_round2 = None
        self.state = (0, None)
        self.round_history = []
        self.history.append(self.state)
        return self.state
        
    def step(self, action: int) -> Tuple[Tuple[int, Optional[int]], float, bool, Dict]:
        """
        Execute une action (choix de l'agent).
        
        Args:
            action: 0=Pierre, 1=Feuille, 2=Ciseaux
            
        Returns:
            tuple: (nouvel_état, récompense, terminé, info)
        """
        if self.done:
            raise ValueError("L'épisode est déjà terminé. Appelez reset().")
            
        if action not in self.action_space:
            raise ValueError(f"Action invalide {action}. Actions valides: {self.action_space}")
            
        reward = 0.0
        
        if self.round == 0:
            # Round 1: L'adversaire joue aléatoirement
            self.agent_choice_round1 = action
            self.opponent_choice_round1 = np.random.choice(self.action_space)
            
            # Calculer le résultat du round 1
            round1_reward = self.outcome_matrix[(action, self.opponent_choice_round1)]
            
            # Sauvegarder le round
            self.round_history.append({
                'round': 1,
                'agent_choice': action,
                'opponent_choice': self.opponent_choice_round1,
                'reward': round1_reward
            })
            
            # Passer au round 2
            self.round = 1
            self.state = (1, self.agent_choice_round1)
            reward = round1_reward
            
        else:  # round == 1
            # Round 2: L'adversaire joue le choix de l'agent au round 1
            self.opponent_choice_round2 = self.agent_choice_round1
            
            # Calculer le résultat du round 2
            round2_reward = self.outcome_matrix[(action, self.opponent_choice_round2)]
            
            # Sauvegarder le round
            self.round_history.append({
                'round': 2,
                'agent_choice': action,
                'opponent_choice': self.opponent_choice_round2,
                'reward': round2_reward
            })
            
            # L'épisode est terminé
            self.done = True
            reward = round2_reward
            
        self.episode_reward += reward
        self.episode_steps += 1
        
        # Sauvegarder dans l'historique
        self.history.append({
            'state': (self.round - 1 if self.done else self.round, self.agent_choice_round1),
            'action': action,
            'reward': reward,
            'next_state': self.state,
            'done': self.done
        })
        
        info = {
            'round': self.round + 1 if self.done else self.round,
            'round_history': self.round_history,
            'episode_reward': self.episode_reward,
            'episode_steps': self.episode_steps
        }
        
        return self.state, reward, self.done, info
        
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Affiche l'état actuel de l'environnement.
        
        Args:
            mode: Mode de rendu
        """
        if mode == 'ansi' or mode == 'human':
            output = "\n" + "="*50 + "\n"
            output += "Two Round Rock Paper Scissors\n"
            output += "="*50 + "\n\n"
            
            # Afficher l'historique des rounds
            for round_info in self.round_history:
                round_num = round_info['round']
                agent_choice = round_info['agent_choice']
                opponent_choice = round_info['opponent_choice']
                reward = round_info['reward']
                
                output += f"Round {round_num}:\n"
                output += f"  Agent: {self.action_symbols[agent_choice]} {self.action_names[agent_choice]}\n"
                output += f"  Adversaire: {self.action_symbols[opponent_choice]} {self.action_names[opponent_choice]}\n"
                
                if reward > 0:
                    output += f"  Résultat: VICTOIRE (+1)\n"
                elif reward < 0:
                    output += f"  Résultat: DÉFAITE (-1)\n"
                else:
                    output += f"  Résultat: ÉGALITÉ (0)\n"
                output += "\n"
                
            # Afficher l'état actuel
            if not self.done:
                output += f"Round actuel: {self.round + 1}\n"
                if self.round == 0:
                    output += "L'adversaire jouera aléatoirement\n"
                else:
                    output += f"L'adversaire jouera: {self.action_symbols[self.agent_choice_round1]} {self.action_names[self.agent_choice_round1]}\n"
                output += "\nChoisissez votre action:\n"
                output += "  0: ✊ Pierre\n"
                output += "  1: ✋ Feuille\n"
                output += "  2: ✌ Ciseaux\n"
            else:
                output += "Partie terminée!\n"
                output += f"Score total: {self.episode_reward:+.0f}\n"
                
            if mode == 'human':
                print(output)
            return output
            
        elif mode == 'rgb_array':
            # Créer une image simple
            img = np.ones((400, 600, 3)) * 255
            
            # TODO: Implémenter le rendu graphique si nécessaire
            
            return img.astype(np.uint8)
            
    def get_action_space(self) -> List[int]:
        """Retourne la liste des actions possibles."""
        return self.action_space
        
    def get_observation_space(self) -> List[Tuple[int, Optional[int]]]:
        """Retourne l'espace des observations."""
        # États possibles: (round, choix_agent_round1)
        states = [(0, None)]  # État initial
        for choice in self.action_space:
            states.append((1, choice))  # États du round 2
        return states
        
    def get_state_representation(self) -> str:
        """Retourne une représentation textuelle de l'état."""
        round_num, choice = self.state
        if choice is None:
            return f"Round 1 (début)"
        else:
            return f"Round 2 (choix round 1: {self.action_names[choice]})"
            
    def get_optimal_strategy(self) -> Dict[Tuple[int, Optional[int]], int]:
        """
        Retourne la stratégie optimale.
        
        La stratégie optimale est:
        - Round 1: Jouer Pierre (0) ou n'importe quoi avec probabilité égale
        - Round 2: Si on a joué Pierre au round 1 → jouer Feuille
                   Si on a joué Feuille au round 1 → jouer Ciseaux
                   Si on a joué Ciseaux au round 1 → jouer Pierre
        """
        return {
            (0, None): 0,  # Round 1: n'importe quel choix
            (1, 0): 1,     # Round 2 après Pierre: jouer Feuille
            (1, 1): 2,     # Round 2 après Feuille: jouer Ciseaux
            (1, 2): 0      # Round 2 après Ciseaux: jouer Pierre
        } 