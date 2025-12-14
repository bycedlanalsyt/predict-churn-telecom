# Prédicteur de Churn – Télécom

> Application de prédiction de churn client utilisant le Machine Learning (XGBoost) avec interface Streamlit

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Table des matières

- [À propos](#-à-propos)
- [Fonctionnalités](#-fonctionnalités)
- [Technologies utilisées](#-technologies-utilisées)
- [Structure du projet](#-structure-du-projet)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Modèle Machine Learning](#-modèle-machine-learning)
- [Aperçu de l'application](#-aperçu-de-lapplication)
- [Auteur](#-auteur)

## À propos

Ce projet est une application complète de prédiction de churn (résiliation) client pour une entreprise de télécommunications. Il combine l'analyse de données, le machine learning et une interface web interactive pour permettre aux utilisateurs de prédire le risque de churn d'un client en fonction de ses caractéristiques.

### Objectifs

- **Prédire le churn** : Identifier les clients à risque de résiliation
- **Interface intuitive** : Application web moderne et facile à utiliser
- **Modèle optimisé** : Utilisation de XGBoost avec GridSearchCV pour une performance maximale
- **Visualisation** : Graphiques et métriques pour comprendre les prédictions

## Fonctionnalités

### Application Streamlit

- 🎨 **Interface moderne** : Design premium avec animations et thème personnalisé
- 📊 **Visualisations** : Graphiques de probabilités et métriques détaillées
- 📥 **Export de données** : Téléchargement des prédictions en format CSV
- ⚡ **Temps réel** : Prédictions instantanées avec indicateur de chargement
- 📱 **Responsive** : Interface adaptée à tous les écrans

### Modèle Machine Learning

- 🚀 **XGBoost optimisé** : Modèle entraîné avec GridSearchCV
- 📈 **Haute précision** : Performance optimale sur les données de test
- 🔄 **Pipeline complet** : Preprocessing automatique (StandardScaler + OneHotEncoder)
- 💾 **Sauvegarde** : Modèle persisté pour utilisation en production

## Technologie utilisée

### Machine Learning
- **XGBoost** : Algorithme de gradient boosting
- **Scikit-learn** : Preprocessing et évaluation
- **Pandas** : Manipulation de données
- **NumPy** : Calculs numériques

### Application Web
- **Streamlit** : Framework pour applications web interactives
- **HTML/CSS** : Personnalisation de l'interface

### Analyse de données
- **Jupyter Notebook** : Analyse exploratoire et développement
- **Matplotlib/Seaborn** : Visualisations

## Structure du projet

```
Project_Data_Sciences/
│
├── Churn_app/                    # Application Streamlit
│   ├── app.py                   # Application principale
│   └── xgb_churn_model.pkl     # Modèle entraîné (à générer)
│
├── notebooks/                    # Notebooks Jupyter
│   └── DataSciences.ipynb       # Analyse complète et entraînement
│
├── data/                         # Données
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── Images_Churn/                # Images de documentation
│
├── .gitignore                   # Fichiers à ignorer par Git
├── README.md                    # Documentation du projet
└── requirements.txt             # Dépendances Python (à créer)
```

## Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Étapes d'installation

1. **Cloner le repository** (ou télécharger le projet)
   ```bash
   git clone <url-du-repo>
   cd Project_Data_Sciences
   ```

2. **Créer un environnement virtuel** (recommandé)
   ```bash
   python -m venv venv
   
   # Sur Windows
   venv\Scripts\activate
   
   # Sur Linux/Mac
   source venv/bin/activate
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```
   ```bash
   pip install streamlit pandas numpy scikit-learn xgboost joblib matplotlib seaborn
   ```

4. **Générer le modèle** (si non présent)
   - Ouvrir le notebook `notebooks/DataSciences.ipynb`
   - Exécuter toutes les cellules jusqu'à la sauvegarde du modèle
   - Le fichier `xgb_churn_model.pkl` sera créé dans `Churn_app/`

##Utilisation

### Lancer l'application Streamlit

1. **Naviguer vers le dossier de l'application**
   ```bash
   cd Churn_app
   ```

2. **Lancer Streamlit**
   ```bash
   streamlit run app.py
   ```

3. **Accéder à l'application**
   - L'application s'ouvrira automatiquement dans votre navigateur
   - URL par défaut : `http://localhost:8501`

### Utiliser l'application

1. **Remplir les paramètres client** dans la barre latérale :
   - Informations personnelles (Genre, Senior Citizen, Partenaire, etc.)
   - Services téléphoniques
   - Services internet
   - Contrat & facturation
   - Frais & ancienneté

2. **Cliquer sur " Prévoir le Churn"**

3. **Consulter les résultats** :
   - Probabilité de churn
   - Recommandations
   - Graphiques de visualisation
   - Métriques détaillées

4. **Télécharger le rapport** (optionnel) :
   - Cliquer sur "Télécharger le Rapport (CSV)"
   - Le rapport contient toutes les informations du client et la prédiction

## Modèle Machine Learning

### Caractéristiques du modèle

- **Algorithme** : XGBoost Classifier
- **Optimisation** : GridSearchCV
- **Preprocessing** :
  - StandardScaler pour les variables numériques
  - OneHotEncoder pour les variables catégorielles
- **Features** : 19 caractéristiques client
- **Métrique d'évaluation** : Accuracy, Precision, Recall, F1-Score

### Variables utilisées

**Numériques** :
- `SeniorCitizen` : Statut senior (0 ou 1)
- `tenure` : Ancienneté en mois
- `MonthlyCharges` : Frais mensuels
- `TotalCharges` : Frais totaux

**Catégorielles** :
- `gender`, `Partner`, `Dependents`
- `PhoneService`, `MultipleLines`
- `InternetService`, `OnlineSecurity`, `OnlineBackup`
- `DeviceProtection`, `TechSupport`
- `StreamingTV`, `StreamingMovies`
- `Contract`, `PaperlessBilling`, `PaymentMethod`

### Performance

Le modèle a été entraîné et optimisé pour maximiser la précision. Les métriques de performance sont disponibles dans le notebook `DataSciences.ipynb`.

## Aperçu de l'application

### Interface principale

- **Header animé** avec gradient teal/bleu
- **Sidebar organisée** en sections repliables
- **Cartes de résultats** avec design moderne
- **Graphiques interactifs** pour visualiser les probabilités

### Résultats de prédiction

- **Carte verte** : Client fidèle (faible risque de churn)
- **Carte rouge** : Risque de churn élevé
- **Barre de progression** : Probabilité visuelle
- **Recommandations** : Actions suggérées selon le résultat

## Développement

### Modifier le modèle

1. Ouvrir `notebooks/DataSciences.ipynb`
2. Modifier les hyperparamètres ou l'algorithme
3. Réentraîner le modèle
4. Sauvegarder le nouveau modèle dans `Churn_app/`

### Personnaliser l'interface

- Modifier les couleurs dans `app.py` (variables `TEAL_PRIMARY`, etc.)
- Ajuster le CSS dans la section `<style>` de `app.py`
- Ajouter de nouvelles fonctionnalités dans les sections appropriées

## 📝 Notes importantes

- ⚠️ Le fichier `xgb_churn_model.pkl` doit être présent dans `Churn_app/` pour que l'application fonctionne
- 📊 Les données d'entraînement doivent être dans `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- 🔄 Le modèle peut être réentraîné en exécutant le notebook complet

## 👤 Auteur

**Cédric BOIMIN**

- Développeur de l'application Prédicteur de Churn

## License

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## Remerciements

- Dataset : [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Streamlit pour le framework web
- XGBoost pour l'algorithme de machine learning
- Scikit-learn pour les outils de preprocessing

## Support

Pour toute question ou problème :
- Ouvrir une issue sur le repository
- Contacter l'auteur : Cédric BOIMIN

---

**Made by Cédric BOIMIN — Data Analyst**
```



