"""
Intégration des environnements secrets
Wrapper pour utiliser les environnements secrets avec notre framework
"""

import sys
import os
from typing import Any, Tuple, Dict, List, Optional
import numpy as np
from .base_environment import BaseEnvironment

# Ajouter le chemin vers le wrapper des environnements secrets
SECRET_ENVS_PATH = os.path.join(os.path.dirname(__file__), 
                                '../../../[Projet] Secret envs 0_1_2 and 3 Python wrapper')
if os.path.exists(SECRET_ENVS_PATH):
    sys.path.append(SECRET_ENVS_PATH)
    try:
        from secret_envs_wrapper import SecretEnv0, SecretEnv1, SecretEnv2, SecretEnv3
        SECRET_ENVS_AVAILABLE = True
    except ImportError:
        print("Attention: Impossible d'importer les environnements secrets")
        SECRET_ENVS_AVAILABLE = False
else:
    SECRET_ENVS_AVAILABLE = False


class SecretEnvironmentWrapper(BaseEnvironment):
    """
    Wrapper pour les environnements secrets.
    Adapte l'interface des environnements secrets à notre framework.
    """
    
    def __init__(self, env_class, env_name: str):
        super().__init__()
        
        if not SECRET_ENVS_AVAILABLE:
            raise ImportError("Les environnements secrets ne sont pas disponibles. "
                            "Vérifiez le chemin et l'installation.")
        
        self.env_name = env_name
        self.secret_env = env_class()
        
        # Obtenir les espaces d'action et d'observation
        self._setup_spaces()
        
    def _setup_spaces(self):
        """Configure les espaces d'action et d'observation."""
        # Essayer d'obtenir les informations depuis l'environnement secret
        if hasattr(self.secret_env, 'action_space'):
            if hasattr(self.secret_env.action_space, 'n'):
                self.action_space = list(range(self.secret_env.action_space.n))
            else:
                # Assumer un espace discret
                self.action_space = list(range(4))  # Par défaut
        else:
            # Utiliser une valeur par défaut
            self.action_space = list(range(4))
            
        # Pour l'espace d'observation, on le découvrira pendant l'exécution
        self.observation_space = None
        
    def reset(self) -> Any:
        """Réinitialise l'environnement."""
        super().reset()
        self.state = self.secret_env.reset()
        self.history.append(self.state)
        return self.state
        
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Execute une action dans l'environnement."""
        if self.done:
            raise ValueError("L'épisode est déjà terminé. Appelez reset().")
            
        # Sauvegarder l'état précédent
        old_state = self.state
        
        # Exécuter l'action dans l'environnement secret
        self.state, reward, self.done, info = self.secret_env.step(action)
        
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
        
        # Ajouter nos propres informations
        info['episode_reward'] = self.episode_reward
        info['episode_steps'] = self.episode_steps
        
        return self.state, reward, self.done, info
        
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Affiche l'état actuel de l'environnement."""
        if hasattr(self.secret_env, 'render'):
            return self.secret_env.render(mode)
        else:
            # Affichage basique si render n'est pas disponible
            output = f"\n{self.env_name}\n"
            output += f"État actuel: {self.state}\n"
            output += f"Récompense totale: {self.episode_reward:.2f}\n"
            output += f"Étapes: {self.episode_steps}\n"
            
            if mode == 'human':
                print(output)
            return output
            
    def get_action_space(self) -> List[Any]:
        """Retourne la liste des actions possibles."""
        return self.action_space
        
    def get_observation_space(self) -> Any:
        """Retourne l'espace des observations."""
        if self.observation_space is not None:
            return self.observation_space
        else:
            # Retourner une description
            return f"Espace d'observation de {self.env_name} (à découvrir)"
            
    def close(self):
        """Ferme l'environnement."""
        if hasattr(self.secret_env, 'close'):
            self.secret_env.close()
        super().close()


# Classes spécifiques pour chaque environnement secret
class SecretEnv0Wrapper(SecretEnvironmentWrapper):
    """Wrapper pour Secret Environment 0."""
    
    def __init__(self):
        if SECRET_ENVS_AVAILABLE:
            super().__init__(SecretEnv0, "Secret Environment 0")
        else:
            raise ImportError("SecretEnv0 non disponible")


class SecretEnv1Wrapper(SecretEnvironmentWrapper):
    """Wrapper pour Secret Environment 1."""
    
    def __init__(self):
        if SECRET_ENVS_AVAILABLE:
            super().__init__(SecretEnv1, "Secret Environment 1")
        else:
            raise ImportError("SecretEnv1 non disponible")


class SecretEnv2Wrapper(SecretEnvironmentWrapper):
    """Wrapper pour Secret Environment 2."""
    
    def __init__(self):
        if SECRET_ENVS_AVAILABLE:
            super().__init__(SecretEnv2, "Secret Environment 2")
        else:
            raise ImportError("SecretEnv2 non disponible")


class SecretEnv3Wrapper(SecretEnvironmentWrapper):
    """Wrapper pour Secret Environment 3."""
    
    def __init__(self):
        if SECRET_ENVS_AVAILABLE:
            super().__init__(SecretEnv3, "Secret Environment 3")
        else:
            raise ImportError("SecretEnv3 non disponible")


# Fonction utilitaire pour obtenir tous les environnements secrets
def get_secret_environments():
    """Retourne une liste des environnements secrets disponibles."""
    envs = {}
    
    if SECRET_ENVS_AVAILABLE:
        try:
            envs['SecretEnv0'] = SecretEnv0Wrapper
            envs['SecretEnv1'] = SecretEnv1Wrapper
            envs['SecretEnv2'] = SecretEnv2Wrapper
            envs['SecretEnv3'] = SecretEnv3Wrapper
        except Exception as e:
            print(f"Erreur lors du chargement des environnements secrets: {e}")
            
    return envs 