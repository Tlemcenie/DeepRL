#!/usr/bin/env python3
"""
Script de test rapide pour une combinaison algorithme-environnement
Usage: python quick_test.py --algorithm QLearning --environment GridWorld --episodes 100
"""

import sys
import os
import argparse
import time
from typing import Dict, Any

# Ajouter le projet au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.environments import *
from src.algorithms import *

# Configuration des algorithmes
ALGORITHMS = {
    'PolicyIteration': PolicyIteration,
    'ValueIteration': ValueIteration,
    'OnPolicyMonteCarlo': OnPolicyMonteCarlo,
    'OffPolicyMonteCarlo': OffPolicyMonteCarlo,
    'QLearning': QLearning,
    'Sarsa': Sarsa,
    'ExpectedSarsa': ExpectedSarsa,
    'DynaQ': DynaQ,
    'DynaQPlus': DynaQPlus
}

# Configuration des environnements
ENVIRONMENTS = {
    'LineWorld': LineWorld,
    'GridWorld': GridWorld,
    'TwoRoundRockPaperScissors': TwoRoundRockPaperScissors,
    'MontyHallLevel1': MontyHallLevel1,
    'MontyHallLevel2': MontyHallLevel2
}


def quick_test(algorithm_name: str, 
               environment_name: str,
               episodes: int = 500,
               alpha: float = 0.1,
               epsilon: float = 0.1,
               gamma: float = 0.99,
               verbose: bool = True) -> Dict[str, Any]:
    """
    Test rapide d'un algorithme sur un environnement.
    
    Args:
        algorithm_name: Nom de l'algorithme
        environment_name: Nom de l'environnement
        episodes: Nombre d'épisodes d'entraînement
        alpha: Learning rate
        epsilon: Epsilon pour epsilon-greedy
        gamma: Facteur de discount
        verbose: Affichage détaillé
    
    Returns:
        Dictionnaire avec les résultats
    """
    if algorithm_name not in ALGORITHMS:
        raise ValueError(f"Algorithme '{algorithm_name}' non reconnu. "
                        f"Disponibles: {list(ALGORITHMS.keys())}")
    
    if environment_name not in ENVIRONMENTS:
        raise ValueError(f"Environnement '{environment_name}' non reconnu. "
                        f"Disponibles: {list(ENVIRONMENTS.keys())}")
    
    print("="*60)
    print(f"🚀 TEST RAPIDE")
    print("="*60)
    print(f"Algorithme: {algorithm_name}")
    print(f"Environnement: {environment_name}")
    print(f"Épisodes: {episodes}")
    print(f"Paramètres: alpha={alpha}, epsilon={epsilon}, gamma={gamma}")
    print("="*60)
    
    # Créer l'environnement
    env_class = ENVIRONMENTS[environment_name]
    env = env_class()
    
    print(f"\n🎮 Environnement créé: {environment_name}")
    print(f"   Actions disponibles: {env.get_action_space()}")
    
    # Préparer les paramètres de l'algorithme
    algo_class = ALGORITHMS[algorithm_name]
    params = {
        'gamma': gamma
    }
    
    # Paramètres spécifiques selon le type d'algorithme
    if algorithm_name in ['QLearning', 'Sarsa', 'ExpectedSarsa', 'DynaQ', 'DynaQPlus']:
        params.update({
            'episodes': episodes,
            'alpha': alpha,
            'epsilon': epsilon
        })
    elif algorithm_name in ['OnPolicyMonteCarlo', 'OffPolicyMonteCarlo']:
        params.update({
            'episodes': episodes,
            'epsilon': epsilon
        })
    elif algorithm_name in ['PolicyIteration', 'ValueIteration']:
        params.update({
            'theta': 1e-6
        })
    
    # Paramètres spéciaux pour les algorithmes de planification
    if algorithm_name == 'DynaQ':
        params['planning_steps'] = 5
    elif algorithm_name == 'DynaQPlus':
        params['planning_steps'] = 5
        params['kappa'] = 0.001
    
    print(f"\n🔧 Création de l'algorithme avec paramètres: {params}")
    
    # Créer et entraîner l'algorithme
    start_time = time.time()
    try:
        algo = algo_class(env, **params)
        
        print(f"\n🎯 Début de l'entraînement...")
        training_results = algo.train(verbose=verbose)
        training_time = time.time() - start_time
        
        print(f"\n✅ Entraînement terminé en {training_time:.2f} secondes")
        
        # Évaluer la politique
        print(f"\n📊 Évaluation de la politique...")
        eval_start = time.time()
        evaluation_results = algo.evaluate_policy(num_episodes=100)
        eval_time = time.time() - eval_start
        
        print(f"✅ Évaluation terminée en {eval_time:.2f} secondes")
        
        # Afficher les résultats
        print(f"\n" + "="*60)
        print(f"📈 RÉSULTATS")
        print(f"="*60)
        print(f"Récompense moyenne: {evaluation_results['mean_reward']:.3f}")
        print(f"Écart-type: {evaluation_results['std_reward']:.3f}")
        print(f"Récompense min: {evaluation_results['min_reward']:.3f}")
        print(f"Récompense max: {evaluation_results['max_reward']:.3f}")
        print(f"Longueur moyenne d'épisode: {evaluation_results['mean_episode_length']:.1f}")
        print(f"Taux de réussite: {evaluation_results['success_rate']*100:.1f}%")
        print(f"Temps d'entraînement: {training_time:.2f}s")
        print(f"Temps d'évaluation: {eval_time:.2f}s")
        print(f"="*60)
        
        # Démonstration de la politique (optionnel)
        if verbose and hasattr(algo, 'demonstrate_policy'):
            print(f"\n🎭 Voulez-vous voir une démonstration de la politique? (y/N)")
            response = input().strip().lower()
            if response == 'y':
                print(f"\n🎭 DÉMONSTRATION DE LA POLITIQUE")
                try:
                    algo.demonstrate_policy(num_steps=10, delay=0.5)
                except Exception as e:
                    print(f"⚠️ Erreur pendant la démonstration: {e}")
        
        # Retourner les résultats
        results = {
            'algorithm': algorithm_name,
            'environment': environment_name,
            'parameters': params,
            'training_time': training_time,
            'evaluation_time': eval_time,
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'success': True
        }
        
        return results
        
    except Exception as e:
        training_time = time.time() - start_time
        print(f"\n❌ ERREUR pendant l'entraînement: {e}")
        
        import traceback
        if verbose:
            traceback.print_exc()
        
        return {
            'algorithm': algorithm_name,
            'environment': environment_name,
            'parameters': params,
            'training_time': training_time,
            'error': str(e),
            'success': False
        }


def interactive_test():
    """Mode interactif pour choisir algorithme et environnement."""
    print("="*60)
    print("🎮 MODE INTERACTIF")
    print("="*60)
    
    # Choisir l'algorithme
    print("\nAlgorithmes disponibles:")
    for i, algo in enumerate(ALGORITHMS.keys(), 1):
        print(f"  {i}. {algo}")
    
    while True:
        try:
            choice = int(input(f"\nChoisissez un algorithme (1-{len(ALGORITHMS)}): ")) - 1
            algorithm_name = list(ALGORITHMS.keys())[choice]
            break
        except (ValueError, IndexError):
            print("❌ Choix invalide, réessayez.")
    
    # Choisir l'environnement
    print(f"\nEnvironnements disponibles:")
    for i, env in enumerate(ENVIRONMENTS.keys(), 1):
        print(f"  {i}. {env}")
    
    while True:
        try:
            choice = int(input(f"\nChoisissez un environnement (1-{len(ENVIRONMENTS)}): ")) - 1
            environment_name = list(ENVIRONMENTS.keys())[choice]
            break
        except (ValueError, IndexError):
            print("❌ Choix invalide, réessayez.")
    
    # Paramètres
    episodes = int(input("\nNombre d'épisodes (défaut: 500): ") or "500")
    alpha = float(input("Learning rate alpha (défaut: 0.1): ") or "0.1")
    epsilon = float(input("Epsilon (défaut: 0.1): ") or "0.1")
    
    print(f"\n🚀 Lancement du test...")
    return quick_test(algorithm_name, environment_name, episodes, alpha, epsilon)


def main():
    """Fonction principale avec interface en ligne de commande."""
    parser = argparse.ArgumentParser(description='Test rapide d\'un algorithme sur un environnement')
    parser.add_argument('--algorithm', '-a', 
                       choices=list(ALGORITHMS.keys()),
                       help='Algorithme à tester')
    parser.add_argument('--environment', '-e',
                       choices=list(ENVIRONMENTS.keys()),
                       help='Environnement à tester')
    parser.add_argument('--episodes', type=int, default=500,
                       help='Nombre d\'épisodes d\'entraînement')
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='Learning rate')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Epsilon pour epsilon-greedy')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Facteur de discount')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Mode silencieux')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Mode interactif')
    
    args = parser.parse_args()
    
    if args.interactive or (not args.algorithm or not args.environment):
        # Mode interactif
        try:
            results = interactive_test()
        except KeyboardInterrupt:
            print("\n👋 Au revoir!")
            return 0
    else:
        # Mode ligne de commande
        try:
            results = quick_test(
                algorithm_name=args.algorithm,
                environment_name=args.environment,
                episodes=args.episodes,
                alpha=args.alpha,
                epsilon=args.epsilon,
                gamma=args.gamma,
                verbose=not args.quiet
            )
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return 1
    
    if results['success']:
        print(f"\n🎉 Test réussi!")
        eval_results = results['evaluation_results']
        print(f"🏆 Score final: {eval_results['mean_reward']:.3f} ± {eval_results['std_reward']:.3f}")
    else:
        print(f"\n💥 Test échoué: {results.get('error', 'Erreur inconnue')}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 