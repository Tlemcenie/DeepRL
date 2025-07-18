#!/usr/bin/env python3
"""
Script pour tester tous les algorithmes sur tous les environnements
Usage: python test_all_algorithms.py --episodes 500 --runs 3
"""

import sys
import os
import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Ajouter le projet au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.environments import *
from src.algorithms import *
from src.utils import ExperimentRunner, ResultsAnalyzer

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    print("⚠️ Matplotlib/Seaborn/NumPy non disponibles. Graphiques désactivés.")
    PLOTTING_AVAILABLE = False
    import numpy as np

# Configuration des algorithmes
ALGORITHM_CONFIGS = {
    # Dynamic Programming
    'PolicyIteration': {
        'class': PolicyIteration,
        'params': {'gamma': 0.99, 'theta': 1e-6},
        'category': 'Dynamic Programming'
    },
    'ValueIteration': {
        'class': ValueIteration,
        'params': {'gamma': 0.99, 'theta': 1e-6},
        'category': 'Dynamic Programming'
    },
    
    # Monte Carlo
    'OnPolicyMonteCarlo': {
        'class': OnPolicyMonteCarlo,
        'params': {'episodes': 1000, 'epsilon': 0.1, 'gamma': 0.99},
        'category': 'Monte Carlo'
    },
    'OffPolicyMonteCarlo': {
        'class': OffPolicyMonteCarlo,
        'params': {'episodes': 1000, 'epsilon': 0.1, 'gamma': 0.99},
        'category': 'Monte Carlo'
    },
    
    # Temporal Difference
    'QLearning': {
        'class': QLearning,
        'params': {'episodes': 1000, 'alpha': 0.1, 'epsilon': 0.1, 'gamma': 0.99},
        'category': 'Temporal Difference'
    },
    'Sarsa': {
        'class': Sarsa,
        'params': {'episodes': 1000, 'alpha': 0.1, 'epsilon': 0.1, 'gamma': 0.99},
        'category': 'Temporal Difference'
    },
    'ExpectedSarsa': {
        'class': ExpectedSarsa,
        'params': {'episodes': 1000, 'alpha': 0.1, 'epsilon': 0.1, 'gamma': 0.99},
        'category': 'Temporal Difference'
    },
    
    # Planning
    'DynaQ': {
        'class': DynaQ,
        'params': {'episodes': 500, 'alpha': 0.1, 'epsilon': 0.1, 'gamma': 0.99, 'planning_steps': 5},
        'category': 'Planning'
    },
    'DynaQPlus': {
        'class': DynaQPlus,
        'params': {'episodes': 500, 'alpha': 0.1, 'epsilon': 0.1, 'gamma': 0.99, 'planning_steps': 5, 'kappa': 0.001},
        'category': 'Planning'
    }
}

# Environnements disponibles
ENVIRONMENTS = [
    ('LineWorld', LineWorld),
    ('GridWorld', GridWorld),
    ('TwoRoundRockPaperScissors', TwoRoundRockPaperScissors),
    ('MontyHallLevel1', MontyHallLevel1),
    ('MontyHallLevel2', MontyHallLevel2)
]


def test_all_algorithms(episodes: Optional[int] = None,
                       num_runs: int = 5,
                       save_results: bool = True,
                       create_visualizations: bool = True,
                       algorithms_to_test: Optional[List[str]] = None,
                       environments_to_test: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Teste tous les algorithmes sur tous les environnements.
    
    Args:
        episodes: Nombre d'épisodes d'entraînement (override)
        num_runs: Nombre d'exécutions par combinaison
        save_results: Si True, sauvegarde les résultats
        create_visualizations: Si True, crée des graphiques
        algorithms_to_test: Liste des algorithmes à tester (None = tous)
        environments_to_test: Liste des environnements à tester (None = tous)
    
    Returns:
        Dictionnaire contenant tous les résultats
    """
    print("="*80)
    print("🚀 TEST COMPLET DE TOUS LES ALGORITHMES")
    print("="*80)
    
    # Filtrer les algorithmes et environnements
    algorithms = {k: v for k, v in ALGORITHM_CONFIGS.items() 
                 if algorithms_to_test is None or k in algorithms_to_test}
    
    environments = [(name, cls) for name, cls in ENVIRONMENTS 
                   if environments_to_test is None or name in environments_to_test]
    
    print(f"Algorithmes à tester: {list(algorithms.keys())}")
    print(f"Environnements à tester: {[name for name, _ in environments]}")
    print(f"Nombre d'exécutions par combinaison: {num_runs}")
    print(f"Total de combinaisons: {len(algorithms)} × {len(environments)} = {len(algorithms) * len(environments)}")
    print(f"Total d'exécutions: {len(algorithms) * len(environments) * num_runs}")
    
    if episodes:
        print(f"Épisodes d'entraînement (override): {episodes}")
    
    print("\n" + "="*80)
    
    # Créer le gestionnaire d'expérimentations
    runner = ExperimentRunner(results_dir="all_algorithms_results")
    
    # Structure des résultats
    all_results = {
        'algorithms': {},
        'environments': list(env[0] for env in environments),
        'performance_matrix': {},
        'summary': {},
        'timestamp': datetime.now().isoformat(),
        'settings': {
            'num_runs': num_runs,
            'episodes_override': episodes
        }
    }
    
    # Résultats détaillés
    total_combinations = len(algorithms) * len(environments)
    current_combination = 0
    
    # Tester chaque combinaison algorithme-environnement
    for algo_name, algo_config in algorithms.items():
        print(f"\n{'🔥' * 60}")
        print(f"ALGORITHME: {algo_name} ({algo_config['category']})")
        print(f"{'🔥' * 60}")
        
        algo_results = {
            'config': algo_config,
            'environments': {},
            'category': algo_config['category']
        }
        
        for env_name, env_class in environments:
            current_combination += 1
            print(f"\n{'='*50}")
            print(f"[{current_combination}/{total_combinations}] {algo_name} sur {env_name}")
            print(f"{'='*50}")
            
            # Préparer les paramètres
            params = algo_config['params'].copy()
            if episodes:
                params['episodes'] = episodes
            
            env_results = {
                'runs': [],
                'statistics': {},
                'execution_time': 0,
                'errors': []
            }
            
            # Exécuter plusieurs runs
            run_rewards = []
            run_times = []
            
            for run in range(num_runs):
                print(f"\n🔄 Run {run + 1}/{num_runs}...")
                
                start_time = time.time()
                try:
                    # Exécuter l'expérimentation
                    result = runner.run_single_experiment(
                        algorithm_class=algo_config['class'],
                        env_class=env_class,
                        algorithm_params=params,
                        experiment_name=f"{algo_name}_{env_name}_run{run+1}"
                    )
                    
                    execution_time = time.time() - start_time
                    
                    env_results['runs'].append(result)
                    run_rewards.append(result['evaluation']['mean_reward'])
                    run_times.append(execution_time)
                    
                    print(f"✅ Récompense: {result['evaluation']['mean_reward']:.3f} "
                          f"(temps: {execution_time:.2f}s)")
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    error_msg = f"Run {run + 1}: {str(e)}"
                    env_results['errors'].append(error_msg)
                    print(f"❌ Erreur: {error_msg}")
                    continue
            
            # Calculer les statistiques
            if run_rewards:
                env_results['statistics'] = {
                    'mean_reward': np.mean(run_rewards),
                    'std_reward': np.std(run_rewards),
                    'min_reward': np.min(run_rewards),
                    'max_reward': np.max(run_rewards),
                    'median_reward': np.median(run_rewards),
                    'success_rate': len(run_rewards) / num_runs,
                    'num_successful_runs': len(run_rewards),
                    'mean_execution_time': np.mean(run_times),
                    'total_execution_time': np.sum(run_times)
                }
                
                print(f"\n📊 STATISTIQUES:")
                stats = env_results['statistics']
                print(f"   Récompense: {stats['mean_reward']:.3f} ± {stats['std_reward']:.3f}")
                print(f"   Plage: [{stats['min_reward']:.3f}, {stats['max_reward']:.3f}]")
                print(f"   Médiane: {stats['median_reward']:.3f}")
                print(f"   Réussite: {stats['num_successful_runs']}/{num_runs} ({stats['success_rate']*100:.1f}%)")
                print(f"   Temps moyen: {stats['mean_execution_time']:.2f}s")
            else:
                env_results['statistics'] = {
                    'mean_reward': -float('inf'),
                    'success_rate': 0,
                    'num_successful_runs': 0
                }
                print("❌ Aucun run réussi pour cette combinaison")
            
            algo_results['environments'][env_name] = env_results
        
        all_results['algorithms'][algo_name] = algo_results
    
    # Créer la matrice de performance
    create_performance_matrix(all_results)
    
    # Calculer le résumé global
    calculate_global_summary(all_results)
    
    # Sauvegarder les résultats
    if save_results:
        save_complete_results(all_results)
    
    # Créer les visualisations
    if create_visualizations and PLOTTING_AVAILABLE:
        create_comprehensive_visualizations(all_results)
    
    # Afficher le résumé final
    print_comprehensive_summary(all_results)
    
    return all_results


def create_performance_matrix(results: Dict[str, Any]):
    """Crée une matrice de performance algorithmes vs environnements."""
    print("\n📊 Création de la matrice de performance...")
    
    matrix = {}
    for algo_name, algo_data in results['algorithms'].items():
        matrix[algo_name] = {}
        for env_name in results['environments']:
            if env_name in algo_data['environments']:
                env_stats = algo_data['environments'][env_name]['statistics']
                matrix[algo_name][env_name] = env_stats.get('mean_reward', -float('inf'))
            else:
                matrix[algo_name][env_name] = -float('inf')
    
    results['performance_matrix'] = matrix


def calculate_global_summary(results: Dict[str, Any]):
    """Calcule les statistiques globales."""
    print("\n🧮 Calcul du résumé global...")
    
    # Statistiques par algorithme
    algo_summary = {}
    for algo_name, algo_data in results['algorithms'].items():
        rewards = []
        success_rates = []
        execution_times = []
        
        for env_name, env_data in algo_data['environments'].items():
            stats = env_data['statistics']
            if stats.get('mean_reward', -float('inf')) != -float('inf'):
                rewards.append(stats['mean_reward'])
                success_rates.append(stats['success_rate'])
                if 'mean_execution_time' in stats:
                    execution_times.append(stats['mean_execution_time'])
        
        if rewards:
            algo_summary[algo_name] = {
                'overall_mean_reward': np.mean(rewards),
                'overall_std_reward': np.std(rewards),
                'overall_success_rate': np.mean(success_rates),
                'num_environments_tested': len(rewards),
                'mean_execution_time': np.mean(execution_times) if execution_times else 0,
                'category': algo_data['category']
            }
        else:
            algo_summary[algo_name] = {
                'overall_mean_reward': -float('inf'),
                'overall_success_rate': 0,
                'num_environments_tested': 0,
                'category': algo_data['category']
            }
    
    # Statistiques par environnement
    env_summary = {}
    for env_name in results['environments']:
        rewards = []
        success_rates = []
        
        for algo_name, algo_data in results['algorithms'].items():
            if env_name in algo_data['environments']:
                stats = algo_data['environments'][env_name]['statistics']
                if stats.get('mean_reward', -float('inf')) != -float('inf'):
                    rewards.append(stats['mean_reward'])
                    success_rates.append(stats['success_rate'])
        
        if rewards:
            env_summary[env_name] = {
                'mean_reward_across_algorithms': np.mean(rewards),
                'std_reward_across_algorithms': np.std(rewards),
                'mean_success_rate': np.mean(success_rates),
                'num_algorithms_tested': len(rewards)
            }
    
    # Meilleurs algorithmes
    valid_algos = {k: v for k, v in algo_summary.items() 
                   if v['overall_mean_reward'] != -float('inf')}
    
    if valid_algos:
        best_overall = max(valid_algos.items(), key=lambda x: x[1]['overall_mean_reward'])
        most_consistent = min(valid_algos.items(), key=lambda x: x[1]['overall_std_reward'])
        fastest = min(valid_algos.items(), key=lambda x: x[1]['mean_execution_time'])
        
        results['summary'] = {
            'algorithm_summary': algo_summary,
            'environment_summary': env_summary,
            'best_overall_algorithm': best_overall[0],
            'most_consistent_algorithm': most_consistent[0],
            'fastest_algorithm': fastest[0],
            'total_experiments': sum(v['num_environments_tested'] for v in valid_algos.values()),
            'categories_tested': list(set(v['category'] for v in valid_algos.values()))
        }


def save_complete_results(results: Dict[str, Any]):
    """Sauvegarde tous les résultats."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Résultats complets
    filename_complete = f"all_algorithms_results/complete_comparison_{timestamp}.json"
    os.makedirs("all_algorithms_results", exist_ok=True)
    
    with open(filename_complete, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    # Matrice de performance simple
    filename_matrix = f"all_algorithms_results/performance_matrix_{timestamp}.json"
    with open(filename_matrix, 'w', encoding='utf-8') as f:
        json.dump(results['performance_matrix'], f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Résultats sauvegardés:")
    print(f"   Complet: {filename_complete}")
    print(f"   Matrice: {filename_matrix}")


def create_comprehensive_visualizations(results: Dict[str, Any]):
    """Crée des visualisations complètes."""
    if not PLOTTING_AVAILABLE:
        print("⚠️ Visualisations non disponibles (bibliothèques manquantes)")
        return
    
    print("\n📈 Création des visualisations complètes...")
    
    try:
        # Configuration
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Heatmap de performance
        ax1 = plt.subplot(2, 3, 1)
        create_performance_heatmap(results, ax1)
        
        # 2. Performance par catégorie
        ax2 = plt.subplot(2, 3, 2)
        create_category_comparison(results, ax2)
        
        # 3. Temps d'exécution
        ax3 = plt.subplot(2, 3, 3)
        create_execution_time_plot(results, ax3)
        
        # 4. Performance par environnement
        ax4 = plt.subplot(2, 3, 4)
        create_environment_comparison(results, ax4)
        
        # 5. Taux de réussite
        ax5 = plt.subplot(2, 3, 5)
        create_success_rate_plot(results, ax5)
        
        # 6. Résumé global
        ax6 = plt.subplot(2, 3, 6)
        create_summary_plot(results, ax6)
        
        plt.suptitle('Analyse Complète des Algorithmes RL', fontsize=20, fontweight='bold')
        plt.tight_layout()
        
        # Sauvegarder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"all_algorithms_results/comprehensive_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualisations sauvegardées: {filename}")
        
    except Exception as e:
        print(f"❌ Erreur lors de la création des visualisations: {e}")


def create_performance_heatmap(results: Dict[str, Any], ax):
    """Crée une heatmap de performance."""
    matrix = results['performance_matrix']
    
    algorithms = list(matrix.keys())
    environments = results['environments']
    
    # Créer la matrice numérique
    data = []
    for algo in algorithms:
        row = [matrix[algo][env] for env in environments]
        data.append(row)
    
    data = np.array(data)
    
    # Remplacer les valeurs infinies
    data[data == -np.inf] = np.nan
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto')
    
    # Configurer les axes
    ax.set_xticks(range(len(environments)))
    ax.set_xticklabels(environments, rotation=45, ha='right')
    ax.set_yticks(range(len(algorithms)))
    ax.set_yticklabels(algorithms)
    
    # Ajouter les valeurs
    for i in range(len(algorithms)):
        for j in range(len(environments)):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', 
                       color='white' if data[i, j] < np.nanmean(data) else 'black')
    
    ax.set_title('Matrice de Performance (Récompense Moyenne)')
    plt.colorbar(im, ax=ax)


def create_category_comparison(results: Dict[str, Any], ax):
    """Compare les performances par catégorie d'algorithme."""
    categories = {}
    
    for algo_name, algo_data in results['algorithms'].items():
        category = algo_data['category']
        if category not in categories:
            categories[category] = []
        
        # Calculer la performance moyenne de cet algorithme
        rewards = []
        for env_data in algo_data['environments'].values():
            stats = env_data['statistics']
            if stats.get('mean_reward', -float('inf')) != -float('inf'):
                rewards.append(stats['mean_reward'])
        
        if rewards:
            categories[category].append(np.mean(rewards))
    
    # Créer le box plot
    cat_names = list(categories.keys())
    cat_data = [categories[cat] for cat in cat_names]
    
    ax.boxplot(cat_data, labels=cat_names)
    ax.set_title('Performance par Catégorie d\'Algorithme')
    ax.set_ylabel('Récompense Moyenne')
    ax.tick_params(axis='x', rotation=45)


def create_execution_time_plot(results: Dict[str, Any], ax):
    """Graphique des temps d'exécution."""
    algorithms = []
    times = []
    
    for algo_name, algo_data in results['algorithms'].items():
        algo_times = []
        for env_data in algo_data['environments'].values():
            stats = env_data['statistics']
            if 'mean_execution_time' in stats:
                algo_times.append(stats['mean_execution_time'])
        
        if algo_times:
            algorithms.append(algo_name)
            times.append(np.mean(algo_times))
    
    if algorithms:
        bars = ax.bar(algorithms, times, color='lightblue', edgecolor='navy')
        ax.set_title('Temps d\'Exécution Moyen par Algorithme')
        ax.set_ylabel('Temps (secondes)')
        ax.tick_params(axis='x', rotation=45)
        
        # Ajouter les valeurs sur les barres
        for bar, time_val in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{time_val:.2f}s', ha='center', va='bottom')


def create_environment_comparison(results: Dict[str, Any], ax):
    """Compare la difficulté des environnements."""
    env_data = results['summary'].get('environment_summary', {})
    
    environments = list(env_data.keys())
    mean_rewards = [env_data[env]['mean_reward_across_algorithms'] for env in environments]
    std_rewards = [env_data[env]['std_reward_across_algorithms'] for env in environments]
    
    ax.errorbar(environments, mean_rewards, yerr=std_rewards, 
               marker='o', capsize=5, linewidth=2, markersize=8)
    ax.set_title('Difficulté des Environnements\n(Performance Moyenne Tous Algorithmes)')
    ax.set_ylabel('Récompense Moyenne')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)


def create_success_rate_plot(results: Dict[str, Any], ax):
    """Graphique des taux de réussite."""
    algorithms = []
    success_rates = []
    
    for algo_name, algo_summary in results['summary']['algorithm_summary'].items():
        if algo_summary['overall_success_rate'] > 0:
            algorithms.append(algo_name)
            success_rates.append(algo_summary['overall_success_rate'] * 100)
    
    if algorithms:
        bars = ax.bar(algorithms, success_rates, color='lightgreen', edgecolor='darkgreen')
        ax.set_title('Taux de Réussite par Algorithme')
        ax.set_ylabel('Taux de Réussite (%)')
        ax.set_ylim(0, 100)
        ax.tick_params(axis='x', rotation=45)
        
        # Ajouter les valeurs
        for bar, rate in zip(bars, success_rates):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                   f'{rate:.1f}%', ha='center', va='bottom')


def create_summary_plot(results: Dict[str, Any], ax):
    """Graphique de résumé avec les meilleurs algorithmes."""
    summary = results['summary']
    
    metrics = ['Meilleur Global', 'Plus Consistant', 'Plus Rapide']
    algorithms = [
        summary.get('best_overall_algorithm', 'N/A'),
        summary.get('most_consistent_algorithm', 'N/A'),
        summary.get('fastest_algorithm', 'N/A')
    ]
    
    colors = ['gold', 'silver', 'bronze']
    bars = ax.bar(metrics, [1, 1, 1], color=colors, alpha=0.7)
    
    # Ajouter les noms des algorithmes
    for i, (bar, algo) in enumerate(zip(bars, algorithms)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height()/2.,
               algo, ha='center', va='center', fontweight='bold', fontsize=10)
    
    ax.set_title('Meilleurs Algorithmes par Catégorie')
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])


def print_comprehensive_summary(results: Dict[str, Any]):
    """Affiche un résumé complet des résultats."""
    print("\n" + "="*80)
    print("🎯 RÉSUMÉ COMPLET DES TESTS")
    print("="*80)
    
    summary = results['summary']
    
    print(f"📊 Vue d'ensemble:")
    print(f"   Total d'expérimentations: {summary.get('total_experiments', 0)}")
    print(f"   Catégories testées: {', '.join(summary.get('categories_tested', []))}")
    print(f"   Algorithmes: {len(results['algorithms'])}")
    print(f"   Environnements: {len(results['environments'])}")
    
    print(f"\n🏆 Meilleurs algorithmes:")
    print(f"   Performance globale: {summary.get('best_overall_algorithm', 'N/A')}")
    print(f"   Plus consistant: {summary.get('most_consistent_algorithm', 'N/A')}")
    print(f"   Plus rapide: {summary.get('fastest_algorithm', 'N/A')}")
    
    print(f"\n📋 Détail par algorithme:")
    algo_summary = summary.get('algorithm_summary', {})
    for algo_name, stats in algo_summary.items():
        if stats['overall_mean_reward'] != -float('inf'):
            print(f"   {algo_name:<20}: {stats['overall_mean_reward']:>7.3f} ± {stats['overall_std_reward']:.3f} "
                  f"(réussite: {stats['overall_success_rate']*100:.0f}%, "
                  f"temps: {stats['mean_execution_time']:.2f}s)")
        else:
            print(f"   {algo_name:<20}: ❌ Échec")
    
    print(f"\n🎮 Difficulté des environnements:")
    env_summary = summary.get('environment_summary', {})
    for env_name, stats in env_summary.items():
        print(f"   {env_name:<25}: {stats['mean_reward_across_algorithms']:>7.3f} ± {stats['std_reward_across_algorithms']:.3f}")
    
    print("\n" + "="*80)


def main():
    """Fonction principale avec interface en ligne de commande."""
    parser = argparse.ArgumentParser(description='Teste tous les algorithmes sur tous les environnements')
    parser.add_argument('--episodes', '-e', type=int, default=None,
                       help='Nombre d\'épisodes d\'entraînement (override)')
    parser.add_argument('--runs', '-r', type=int, default=3,
                       help='Nombre d\'exécutions par combinaison')
    parser.add_argument('--algorithms', '-a', nargs='+', 
                       choices=list(ALGORITHM_CONFIGS.keys()),
                       help='Algorithmes spécifiques à tester')
    parser.add_argument('--environments', '-env', nargs='+',
                       choices=[name for name, _ in ENVIRONMENTS],
                       help='Environnements spécifiques à tester')
    parser.add_argument('--no-save', action='store_true',
                       help='Ne pas sauvegarder les résultats')
    parser.add_argument('--no-viz', action='store_true',
                       help='Ne pas créer de visualisations')
    parser.add_argument('--quick', action='store_true',
                       help='Test rapide (moins d\'épisodes et de runs)')
    
    args = parser.parse_args()
    
    # Mode rapide
    if args.quick:
        episodes = 100
        runs = 2
        print("🚀 Mode rapide activé (100 épisodes, 2 runs)")
    else:
        episodes = args.episodes
        runs = args.runs
    
    print("🚀 Démarrage du test complet...")
    print(f"Configuration:")
    print(f"  Épisodes: {episodes if episodes else 'par défaut'}")
    print(f"  Runs par combinaison: {runs}")
    if args.algorithms:
        print(f"  Algorithmes: {args.algorithms}")
    if args.environments:
        print(f"  Environnements: {args.environments}")
    
    try:
        results = test_all_algorithms(
            episodes=episodes,
            num_runs=runs,
            save_results=not args.no_save,
            create_visualizations=not args.no_viz,
            algorithms_to_test=args.algorithms,
            environments_to_test=args.environments
        )
        
        print("\n✅ Test complet terminé avec succès!")
        print(f"🏆 Meilleur algorithme global: {results['summary'].get('best_overall_algorithm', 'N/A')}")
        
    except Exception as e:
        print(f"\n❌ Erreur pendant le test: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 