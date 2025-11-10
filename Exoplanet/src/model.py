"""
Module de modélisation avec ensemble learning pour ExoKeplerAI
Combine LightGBM, CatBoost et XGBoost
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')


class ExoplanetEnsembleClassifier:
    """
    Classificateur d'ensemble combinant LightGBM, CatBoost et XGBoost
    """
    
    def __init__(self, random_state=42):
        """
        Initialise les trois modèles
        
        Args:
            random_state: Seed pour la reproductibilité
        """
        self.random_state = random_state
        self.models = {}
        self.weights = {'lgb': 0.33, 'cat': 0.34, 'xgb': 0.33}  # Poids initiaux égaux
        self.class_names = ['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED']
        
        # Initialiser les modèles
        self._initialize_models()
    
    def _initialize_models(self):
        """
        Initialise les trois modèles avec leurs hyperparamètres
        """
        print("🔧 Initialisation des modèles...")
        
        # LightGBM
        self.models['lgb'] = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            verbose=-1,
            class_weight='balanced'
        )
        
        # CatBoost
        self.models['cat'] = cb.CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=3,
            random_state=self.random_state,
            verbose=False,
            auto_class_weights='Balanced'
        )
        
        # XGBoost
        self.models['xgb'] = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            verbosity=0,
            eval_metric='mlogloss'
        )
        
        print("   LightGBM initialisé")
        print("   CatBoost initialisé")
        print("   XGBoost initialisé")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Entraîne les trois modèles
        
        Args:
            X_train: Features d'entraînement
            y_train: Target d'entraînement
            X_val: Features de validation (optionnel)
            y_val: Target de validation (optionnel)
        """
        print("ENTRAÎNEMENT DES MODÈLES")
        
        # Entraîner LightGBM
        print("Entraînement de LightGBM...")
        if X_val is not None and y_val is not None:
            self.models['lgb'].fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='multi_logloss',
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
        else:
            self.models['lgb'].fit(X_train, y_train)
        print("   LightGBM entraîné")
        
        # Entraîner CatBoost
        print("\nEntraînement de CatBoost...")
        if X_val is not None and y_val is not None:
            self.models['cat'].fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            self.models['cat'].fit(X_train, y_train)
        print("   CatBoost entraîné")
        
        # Entraîner XGBoost
        print("\nEntraînement de XGBoost...")
        if X_val is not None and y_val is not None:
            self.models['xgb'].fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.models['xgb'].fit(X_train, y_train)
        print("   XGBoost entraîné")
        
        print(" ENTRAÎNEMENT TERMINÉ")
    
    def predict_proba(self, X):
        """
        Prédit les probabilités avec ensemble voting
        
        Args:
            X: Features
            
        Returns:
            Probabilités pour chaque classe
        """
        # Obtenir les probabilités de chaque modèle
        proba_lgb = self.models['lgb'].predict_proba(X)
        proba_cat = self.models['cat'].predict_proba(X)
        proba_xgb = self.models['xgb'].predict_proba(X)
        
        # Moyenne pondérée
        proba_ensemble = (
            self.weights['lgb'] * proba_lgb +
            self.weights['cat'] * proba_cat +
            self.weights['xgb'] * proba_xgb
        )
        
        return proba_ensemble
    
    def predict(self, X):
        """
        Prédit les classes avec ensemble voting
        
        Args:
            X: Features
            
        Returns:
            Prédictions de classe
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def evaluate(self, X_test, y_test, detailed=True):
        """
        Évalue les performances du modèle
        
        Args:
            X_test: Features de test
            y_test: Target de test
            detailed: Afficher les détails
            
        Returns:
            Dictionnaire avec les métriques
        """
        print(" ÉVALUATION DES PERFORMANCES")
        
        # Prédictions de l'ensemble
        y_pred_ensemble = self.predict(X_test)
        
        # Métriques globales
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred_ensemble),
            'precision': precision_score(y_test, y_pred_ensemble, average='weighted'),
            'recall': recall_score(y_test, y_pred_ensemble, average='weighted'),
            'f1_score': f1_score(y_test, y_pred_ensemble, average='weighted')
        }
        
        if detailed:
            print(" MODÈLE D'ENSEMBLE")
            print(f"   Accuracy:  {metrics['accuracy']:.4f}")
            print(f"   Precision: {metrics['precision']:.4f}")
            print(f"   Recall:    {metrics['recall']:.4f}")
            print(f"   F1-Score:  {metrics['f1_score']:.4f}")
            
            # Évaluer chaque modèle individuellement
            print("\n PERFORMANCES INDIVIDUELLES")
            
            for name, model in self.models.items():
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                print(f"   {name.upper()}: Accuracy = {acc:.4f}")
            
            # Rapport de classification détaillé
            print("\n RAPPORT DE CLASSIFICATION")
            print(classification_report(
                y_test, y_pred_ensemble,
                target_names=self.class_names,
                digits=4
            ))
            
            # Matrice de confusion
            print(" MATRICE DE CONFUSION")
            cm = confusion_matrix(y_test, y_pred_ensemble)
            print(pd.DataFrame(
                cm,
                index=[f'True {c}' for c in self.class_names],
                columns=[f'Pred {c}' for c in self.class_names]
            ))
        
        
        return metrics
    
    def get_feature_importance(self, feature_names, top_n=20):
        """
        Obtient l'importance des features de chaque modèle
        
        Args:
            feature_names: Noms des features
            top_n: Nombre de top features à retourner
            
        Returns:
            DataFrame avec l'importance des features
        """
        # LightGBM
        importance_lgb = pd.DataFrame({
            'feature': feature_names,
            'importance_lgb': self.models['lgb'].feature_importances_
        })
        
        # CatBoost
        importance_cat = pd.DataFrame({
            'feature': feature_names,
            'importance_cat': self.models['cat'].feature_importances_
        })
        
        # XGBoost
        importance_xgb = pd.DataFrame({
            'feature': feature_names,
            'importance_xgb': self.models['xgb'].feature_importances_
        })
        
        # Fusionner
        importance = importance_lgb.merge(importance_cat, on='feature').merge(importance_xgb, on='feature')
        
        # Moyenne pondérée
        importance['importance_avg'] = (
            self.weights['lgb'] * importance['importance_lgb'] +
            self.weights['cat'] * importance['importance_cat'] +
            self.weights['xgb'] * importance['importance_xgb']
        )
        
        # Trier et retourner top N
        importance = importance.sort_values('importance_avg', ascending=False).head(top_n)
        
        return importance
    
    def save_models(self, directory='models'):
        """
        Sauvegarde les modèles entraînés
        
        Args:
            directory: Répertoire de sauvegarde
        """
        import os
        os.makedirs(directory, exist_ok=True)
        
        print(f"\n Sauvegarde des modèles dans '{directory}/'...")
        
        # Sauvegarder chaque modèle
        joblib.dump(self.models['lgb'], f'{directory}/lightgbm_model.pkl')
        self.models['cat'].save_model(f'{directory}/catboost_model.cbm')
        self.models['xgb'].save_model(f'{directory}/xgboost_model.json')
        
        # Sauvegarder les poids
        joblib.dump(self.weights, f'{directory}/ensemble_weights.pkl')
        
        print("    LightGBM sauvegardé")
        print("    CatBoost sauvegardé")
        print("    XGBoost sauvegardé")
        print("    Poids d'ensemble sauvegardés")
    
    def load_models(self, directory='models'):
        """
        Charge les modèles sauvegardés
        
        Args:
            directory: Répertoire de chargement
        """
        print(f"\n Chargement des modèles depuis '{directory}/'...")
        
        # Charger chaque modèle
        self.models['lgb'] = joblib.load(f'{directory}/lightgbm_model.pkl')
        self.models['cat'] = cb.CatBoostClassifier()
        self.models['cat'].load_model(f'{directory}/catboost_model.cbm')
        self.models['xgb'] = xgb.XGBClassifier()
        self.models['xgb'].load_model(f'{directory}/xgboost_model.json')
        
        # Charger les poids
        self.weights = joblib.load(f'{directory}/ensemble_weights.pkl')
        
        print("    LightGBM chargé")
        print("    CatBoost chargé")
        print("    XGBoost chargé")
        print("    Poids d'ensemble chargés")
    
    def predict_single(self, X):
        """
        Prédit pour une seule observation avec détails
        
        Args:
            X: Features (1 observation)
            
        Returns:
            Dictionnaire avec prédiction et probabilités
        """
        proba = self.predict_proba(X)[0]
        pred_class = np.argmax(proba)
        
        # Obtenir les prédictions individuelles avec conversion sécurisée
        lgb_pred = int(self.models['lgb'].predict(X)[0])
        cat_pred = int(self.models['cat'].predict(X)[0])
        xgb_pred = int(self.models['xgb'].predict(X)[0])
        
        return {
            'prediction': self.class_names[pred_class],
            'prediction_code': int(pred_class),
            'confidence': float(proba[pred_class]),
            'probabilities': {
                'FALSE POSITIVE': float(proba[0]),
                'CANDIDATE': float(proba[1]),
                'CONFIRMED': float(proba[2])
            },
            'individual_predictions': {
                'LightGBM': self.class_names[lgb_pred],
                'CatBoost': self.class_names[cat_pred],
                'XGBoost': self.class_names[xgb_pred]
            }
        }
