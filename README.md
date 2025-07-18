# Projet Deep Reinforcement Learning - 4A IABD

## 📋 Description

Ce projet implémente une bibliothèque complète d'algorithmes d'apprentissage par renforcement et d'environnements de test. Il a été réalisé dans le cadre du cours de Deep Reinforcement Learning de 4A IABD.

## 🚀 Installation

### Prérequis
- Python 3.8+
- pip

### Installation des dépendances
```bash
cd rl_project
pip install -r requirements.txt
```

### Installation en mode développement
```bash
pip install -e .
```

## 📁 Structure du Projet

```
rl_project/
├── src/
│   ├── algorithms/         # Algorithmes RL
│   ├── environments/       # Environnements
│   └── utils/             # Utilitaires
├── experiments/           # Scripts d'expérimentation
├── results/              # Résultats sauvegardés
├── docs/                 # Documentation
├── presentation/         # Slides de présentation
└── Rapport_Projet_RL.ipynb  # Rapport Jupyter
```

## 🎮 Environnements Implémentés

1. **Line World** : Monde linéaire avec récompenses aux extrémités
2. **Grid World** : Grille 2D avec obstacles et objectifs
3. **Two Round Rock Paper Scissors** : Pierre-Feuille-Ciseaux en 2 rounds
4. **Monty Hall Level 1** : Paradoxe de Monty Hall avec 3 portes
5. **Monty Hall Level 2** : Version étendue avec 5 portes

## 🤖 Algorithmes Implémentés

### Dynamic Programming
- Policy Iteration
- Value Iteration

### Monte Carlo Methods
- Monte Carlo ES (Exploring Starts)
- On-policy First-visit Monte Carlo Control
- Off-policy Monte Carlo Control

### Temporal Difference Learning
- SARSA
- Q-Learning
- Expected SARSA

### Planning
- Dyna-Q
- Dyna-Q+

## 🔧 Utilisation

### Lancer toutes les expérimentations
```bash
python main_experiments.py
```

### Exemple d'utilisation simple
```python
from src.environments import LineWorld
from src.algorithms import QLearning

# Créer l'environnement
env = LineWorld(size=7)

# Créer et entraîner l'algorithme
algo = QLearning(env, episodes=1000, alpha=0.1, epsilon=0.1)
results = algo.train()

# Évaluer la politique
eval_results = algo.evaluate_policy(num_episodes=100)
print(f"Récompense moyenne: {eval_results['mean_reward']}")

# Démontrer la politique
algo.demonstrate_policy()
```

### Optimisation des hyperparamètres
```python
from src.utils import HyperparameterTuner

tuner = HyperparameterTuner()
best_params, results = tuner.grid_search(
    algorithm_class=QLearning,
    env_class=LineWorld,
    param_grid={
        'alpha': [0.01, 0.1, 0.5],
        'epsilon': [0.01, 0.1, 0.3],
        'gamma': [0.9, 0.95, 0.99]
    }
)
```

## 📊 Résultats

Les résultats des expérimentations sont sauvegardés dans le dossier `results/` avec :
- Fichiers JSON des résultats
- Modèles entraînés (pickle)
- Graphiques de performance
- Matrice de comparaison

## 📈 Visualisations

Le projet inclut plusieurs outils de visualisation :
- Courbes d'apprentissage
- Comparaison des performances
- Heatmaps des fonctions de valeur
- Visualisation des politiques

## 🧪 Tests

Pour tester un environnement manuellement :
```python
env = LineWorld()
env.play_manual()
```

## 📚 Documentation

Consultez le notebook `Rapport_Projet_RL.ipynb` pour :
- Description détaillée des algorithmes
- Analyse des résultats
- Comparaisons statistiques
- Interprétations

## 👥 Auteurs

Groupe IABD - 4A
- Année : 2024-2025

## 📄 Licence

Ce projet est réalisé dans un cadre académique. 