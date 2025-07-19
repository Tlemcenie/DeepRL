#!/usr/bin/env python3
"""
Test rapide pour vérifier la correction du problème JSON
"""

import sys
import os
sys.path.append(os.getcwd())

from src.algorithms.temporal_difference import QLearning
from src.environments.line_world import LineWorld
from src.utils.experiment_runner import ExperimentRunner

def test_qlearning_fix():
    """Test Q-Learning sur LineWorld pour vérifier le fix."""
    print("🧪 Test de la correction du problème JSON...")
    
    # Créer le runner
    runner = ExperimentRunner(results_dir="test_results")
    
    # Paramètres Q-Learning
    params = {
        'episodes': 100,  # Réduit pour le test
        'alpha': 0.1,
        'epsilon': 0.1,
        'gamma': 0.99,
        'epsilon_decay': 0.995
    }
    
    try:
        # Lancer l'expérimentation
        result = runner.run_single_experiment(
            algorithm_class=QLearning,
            env_class=LineWorld,
            algorithm_params=params,
            experiment_name="test_fix_QLearning_LineWorld"
        )
        
        if 'evaluation' in result:
            print(f"✅ SUCCÈS ! Récompense moyenne: {result['evaluation']['mean_reward']:.3f}")
            return True
        else:
            print(f"❌ Problème: pas d'évaluation dans les résultats")
            return False
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    success = test_qlearning_fix()
    if success:
        print("\n🎉 La correction fonctionne ! Vous pouvez maintenant utiliser le notebook.")
    else:
        print("\n⚠️ Il reste des problèmes à corriger.") 