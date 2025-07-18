# Guide des Tests d'Algorithmes

Ce guide vous explique comment utiliser les différents scripts de test pour évaluer les algorithmes d'apprentissage par renforcement sur vos environnements.

## 📁 Scripts Disponibles

### 1. `quick_test.py` - Test Rapide
**Usage**: Tester rapidement une combinaison algorithme-environnement
```bash
python quick_test.py --algorithm QLearning --environment GridWorld --episodes 100
```

### 2. `test_single_algorithm.py` - Test d'un Algorithme
**Usage**: Tester un algorithme sur tous les environnements
```bash
python test_single_algorithm.py --algorithm QLearning --episodes 1000 --runs 5
```

### 3. `test_all_algorithms.py` - Test Complet
**Usage**: Tester tous les algorithmes sur tous les environnements
```bash
python test_all_algorithms.py --episodes 500 --runs 3
```

## 🎯 Algorithmes Disponibles

| Algorithme | Catégorie | Description |
|------------|-----------|-------------|
| `PolicyIteration` | Dynamic Programming | Itération de politique |
| `ValueIteration` | Dynamic Programming | Itération de valeur |
| `OnPolicyMonteCarlo` | Monte Carlo | Monte Carlo on-policy |
| `OffPolicyMonteCarlo` | Monte Carlo | Monte Carlo off-policy |
| `QLearning` | Temporal Difference | Q-Learning |
| `Sarsa` | Temporal Difference | SARSA |
| `ExpectedSarsa` | Temporal Difference | Expected SARSA |
| `DynaQ` | Planning | Dyna-Q |
| `DynaQPlus` | Planning | Dyna-Q+ |

## 🎮 Environnements Disponibles

| Environnement | Description |
|---------------|-------------|
| `LineWorld` | Monde linéaire avec objectifs aux extrémités |
| `GridWorld` | Grille 2D avec obstacles et récompenses |
| `TwoRoundRockPaperScissors` | Pierre-feuille-ciseaux en 2 rounds |
| `MontyHallLevel1` | Paradoxe de Monty Hall (3 portes) |
| `MontyHallLevel2` | Paradoxe de Monty Hall étendu (5 portes) |

## 🚀 Exemples d'Utilisation

### Test Rapide
```bash
# Test Q-Learning sur GridWorld
python quick_test.py -a QLearning -e GridWorld --episodes 200

# Mode interactif
python quick_test.py --interactive

# Test silencieux
python quick_test.py -a Sarsa -e LineWorld --quiet
```

### Test d'un Algorithme Spécifique
```bash
# Tester Q-Learning sur tous les environnements
python test_single_algorithm.py --algorithm QLearning --episodes 1000 --runs 5

# Avec tuning d'hyperparamètres
python test_single_algorithm.py --algorithm QLearning --tune --runs 3

# Paramètres personnalisés
python test_single_algorithm.py --algorithm QLearning --alpha 0.2 --epsilon 0.05
```

### Test Complet
```bash
# Test complet avec paramètres par défaut
python test_all_algorithms.py

# Test rapide (moins d'épisodes et de runs)
python test_all_algorithms.py --quick

# Test de certains algorithmes seulement
python test_all_algorithms.py --algorithms QLearning Sarsa DynaQ

# Test sur certains environnements seulement
python test_all_algorithms.py --environments GridWorld LineWorld

# Sans sauvegarde ni visualisations
python test_all_algorithms.py --no-save --no-viz
```

## 📊 Résultats et Analyses

### Structure des Résultats
Les résultats sont sauvegardés dans des dossiers organisés :
```
results/
├── single_algorithm_results/     # Résultats des tests d'un algorithme
├── all_algorithms_results/       # Résultats des tests complets
└── experiment_YYYYMMDD_HHMMSS/   # Expérimentations du main_experiments.py
```

### Fichiers Générés
- **JSON** : Résultats détaillés avec toutes les métriques
- **PNG** : Graphiques et visualisations
- **CSV** : Matrices de performance (si applicable)

### Métriques Analysées
- **Récompense moyenne** : Performance principale
- **Écart-type** : Consistance de l'algorithme
- **Taux de réussite** : Pourcentage de runs réussis
- **Temps d'exécution** : Efficacité computationnelle
- **Longueur d'épisode** : Efficacité de la politique

## 🔧 Paramètres Avancés

### Hyperparamètres Communs
| Paramètre | Description | Valeur par défaut |
|-----------|-------------|-------------------|
| `episodes` | Nombre d'épisodes d'entraînement | 500-1000 |
| `alpha` | Learning rate (TD) | 0.1 |
| `epsilon` | Exploration (epsilon-greedy) | 0.1 |
| `gamma` | Facteur de discount | 0.99 |

### Paramètres Spécialisés
- **DynaQ/DynaQ+** : `planning_steps` (défaut: 5)
- **DynaQ+** : `kappa` pour bonus d'exploration (défaut: 0.001)
- **DP** : `theta` pour convergence (défaut: 1e-6)

## 🎭 Mode Démonstration
Certains algorithmes supportent une démonstration visuelle de la politique apprise :
```bash
python quick_test.py -a QLearning -e GridWorld
# Puis répondre 'y' quand demandé
```

## 📈 Tuning d'Hyperparamètres
Pour optimiser automatiquement les hyperparamètres :
```bash
python test_single_algorithm.py --algorithm QLearning --tune
```

Cette fonctionnalité teste plusieurs combinaisons de paramètres et retourne la meilleure configuration.

## 🐛 Dépannage

### Erreurs Communes
1. **Import Error** : Vérifiez que vous êtes dans le bon répertoire
2. **Algorithm/Environment not found** : Vérifiez l'orthographe
3. **Memory Error** : Réduisez le nombre d'épisodes ou de runs

### Performance
- Utilisez `--quick` pour des tests rapides
- Réduisez `--runs` si les tests sont trop longs
- Utilisez `--quiet` pour accélérer l'affichage

## 💡 Conseils d'Utilisation

### Pour Débuter
1. Commencez par `quick_test.py` pour vous familiariser
2. Testez chaque algorithme individuellement avec `test_single_algorithm.py`
3. Une fois satisfait, lancez le test complet

### Pour l'Analyse
1. Sauvegardez toujours les résultats (`--no-save` seulement pour debug)
2. Utilisez plusieurs runs (3-5) pour des résultats fiables
3. Comparez les catégories d'algorithmes (DP vs TD vs MC vs Planning)

### Pour l'Optimisation
1. Utilisez le tuning d'hyperparamètres pour les algorithmes sensibles
2. Adaptez le nombre d'épisodes selon l'environnement
3. Surveillez les temps d'exécution pour l'efficacité

## 📚 Interprétation des Résultats

### Récompense Moyenne
- **Positive** : L'algorithme apprend une politique efficace
- **Négative** : Difficultés d'apprentissage ou environnement difficile
- **Variable** : Comparez avec l'écart-type pour évaluer la consistance

### Taux de Réussite
- **100%** : Algorithme stable sur cet environnement
- **< 100%** : Problèmes potentiels (hyperparamètres, convergence)

### Temps d'Exécution
- Comparez entre algorithmes pour choisir selon vos contraintes
- Les algorithmes DP sont souvent plus rapides mais nécessitent un modèle
- Les algorithmes de planning (Dyna) sont plus lents mais plus flexibles

## 🎯 Workflow Recommandé

1. **Phase d'Exploration** : `quick_test.py` sur diverses combinaisons
2. **Phase d'Analyse** : `test_single_algorithm.py` pour les algorithmes prometteurs
3. **Phase de Comparaison** : `test_all_algorithms.py` pour l'analyse finale
4. **Phase d'Optimisation** : Tuning d'hyperparamètres sur les meilleurs

Bon apprentissage par renforcement ! 🚀 