# Valve Condition Prediction

# 🔧 Predictive Maintenance – Valve Condition Monitoring

## 📌 Contexte du projet
Ce projet s’inscrit dans un contexte industriel de **maintenance prédictive**.  
L’objectif est de **prédire la condition d’une valve hydraulique** pour chaque cycle de production à partir de signaux capteurs (pression, débit, température, vibration, etc.).

Le jeu de données provient du **UCI Machine Learning Repository** :  
👉 https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems

---

## 🎯 Objectif
Construire un **modèle de Machine Learning** capable de prédire si la condition de la valve est :
- **Optimale (100%)**
- **Non optimale**

Pour chaque cycle de production :
- Les **2000 premiers cycles** sont utilisés pour l’entraînement
- Les cycles restants sont utilisés comme **jeu de test final**

---


---

## 📊 Données utilisées

### Capteurs exploités
- **Pression** : `PS1 – PS6`
- **Débit** : `FS1, FS2`
- **Température** : `TS1 – TS4`
- **Vibration** : `VS1`
- **Puissance électrique** : `EPS1`

⚠️ Les variables **CP, CE, SE** ont été exclues :
> Elles décrivent directement l’état du système et introduisent une **fuite d’information (data leakage)**.

---

## ⚙️ Méthodologie

### 1️⃣ Exploration des données
- Analyse des signaux bruts par cycle
- Visualisation des capteurs (pression, débit, température)
- Comparaison **signal brut vs moyenne**

### 2️⃣ Feature Engineering
Les signaux haute fréquence sont résumés **par cycle** à l’aide de statistiques robustes :

- Moyenne
- Écart-type
- Minimum
- Maximum

➡️ Justification :
- Les signaux sont **quasi stationnaires par cycle**
- Les variations intra-cycle sont faibles
- Réduction drastique de la dimension sans perte d’information critique

### 3️⃣ Nettoyage & Prétraitement
- Suppression des valeurs aberrantes extrêmes (méthode IQR)
- Analyse des zéros physiques (pression nulle possible)
- Standardisation des features
- PCA appliquée aux capteurs de pression

### 4️⃣ Modélisation
- Séparation temporelle des données (pas de shuffle)
- Modèles testés :
  - Logistic Regression
  - Random Forest
  - Gradient Boosting

### 5️⃣ Évaluation
- Accuracy
- Precision / Recall
- F1-score
- Matrice de confusion

---

## 📈 Résultats

Exemple de résultats obtenus sur le jeu de test :




