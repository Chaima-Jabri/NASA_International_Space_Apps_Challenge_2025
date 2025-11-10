"""
Module de préprocessing des données Kepler pour ExoKeplerAI
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class KeplerDataPreprocessor:
    """
    Classe pour le préprocessing des données Kepler
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.feature_medians = {}  # Pour stocker les médianes
        self.target_mapping = {
            'CONFIRMED': 2,
            'CANDIDATE': 1,
            'FALSE POSITIVE': 0
        }
        
    def load_data(self, filepath):
        """
        Charge les données Kepler depuis un fichier CSV
        
        Args:
            filepath: Chemin vers le fichier CSV
            
        Returns:
            DataFrame pandas
        """
        print(f"Chargement des données depuis {filepath}...")
        df = pd.read_csv(filepath, comment='#')
        print(f"Données chargées: {df.shape[0]} observations, {df.shape[1]} colonnes")
        return df
    
    def select_features(self, df):
        """
        Sélectionne les features pertinentes pour la classification
        
        Args:
            df: DataFrame pandas
            
        Returns:
            DataFrame avec features sélectionnées
        """
        # Features clés pour la détection d'exoplanètes
        important_features = [
            # Paramètres orbitaux
            'koi_period',           # Période orbitale
            'koi_duration',         # Durée du transit
            'koi_depth',            # Profondeur du transit
            'koi_impact',           # Paramètre d'impact
            
            # Paramètres planétaires
            'koi_prad',             # Rayon planétaire
            'koi_teq',              # Température d'équilibre
            'koi_insol',            # Flux d'insolation
            
            # Paramètres stellaires
            'koi_steff',            # Température stellaire
            'koi_slogg',            # Gravité de surface stellaire
            'koi_srad',             # Rayon stellaire
            
            # Métriques de qualité
            'koi_model_snr',        # Signal-to-Noise Ratio
            'koi_score',            # Score de disposition
            
            # Flags de faux positifs
            'koi_fpflag_nt',        # Not Transit-Like
            'koi_fpflag_ss',        # Stellar Eclipse
            'koi_fpflag_co',        # Centroid Offset
            'koi_fpflag_ec',        # Ephemeris Match
            
            # Autres
            'koi_kepmag',           # Magnitude Kepler
        ]
        
        # Vérifier quelles features sont disponibles
        available_features = [f for f in important_features if f in df.columns]
        
        print(f"Features sélectionnées: {len(available_features)}/{len(important_features)}")
        
        return df[available_features], available_features
    
    def handle_missing_values(self, df, fit=True):
        """
        Gère les valeurs manquantes
        
        Args:
            df: DataFrame pandas
            fit: Si True, fit l'imputer, sinon utilise l'imputer existant
            
        Returns:
            DataFrame sans valeurs manquantes
        """
        print("Traitement des valeurs manquantes...")
        
        # Compter les valeurs manquantes avant
        missing_before = df.isnull().sum().sum()
        
        if fit:
            # Sauvegarder les médianes pour utilisation future
            self.feature_medians = df.median().to_dict()
            # Imputation par la médiane pour les features numériques
            df_imputed = pd.DataFrame(
                self.imputer.fit_transform(df),
                columns=df.columns,
                index=df.index
            )
        else:
            # Utiliser l'imputer déjà fitté
            df_imputed = pd.DataFrame(
                self.imputer.transform(df),
                columns=df.columns,
                index=df.index
            )
        
        missing_after = df_imputed.isnull().sum().sum()
        
        print(f"   Valeurs manquantes: {missing_before} → {missing_after}")
        
        return df_imputed
    
    def create_engineered_features(self, df):
        """
        Crée de nouvelles features par ingénierie
        
        Args:
            df: DataFrame pandas
            
        Returns:
            DataFrame avec features supplémentaires
        """
        print("Création de features engineered...")
        
        df_new = df.copy()
        
        # Ratio rayon planétaire / rayon stellaire
        if 'koi_prad' in df.columns and 'koi_srad' in df.columns:
            df_new['prad_srad_ratio'] = df['koi_prad'] / (df['koi_srad'] * 109.1)  # 109.1 = rayon solaire en rayons terrestres
        
        # Densité relative (approximation)
        if 'koi_prad' in df.columns and 'koi_period' in df.columns:
            df_new['density_proxy'] = df['koi_prad'] / np.sqrt(df['koi_period'])
        
        # Score de confiance composite
        if 'koi_score' in df.columns and 'koi_model_snr' in df.columns:
            df_new['confidence_score'] = df['koi_score'] * np.log1p(df['koi_model_snr'])
        
        # Flag composite de faux positifs
        fp_flags = ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']
        if all(f in df.columns for f in fp_flags):
            df_new['total_fp_flags'] = df[fp_flags].sum(axis=1)
        
        # Zone habitable (approximation simple)
        if 'koi_insol' in df.columns:
            df_new['in_habitable_zone'] = ((df['koi_insol'] >= 0.25) & (df['koi_insol'] <= 4.0)).astype(int)
        
        print(f"   {len(df_new.columns) - len(df.columns)} nouvelles features créées")
        
        return df_new
    
    def normalize_features(self, X_train, X_test=None):
        """
        Normalise les features avec StandardScaler
        
        Args:
            X_train: Features d'entraînement
            X_test: Features de test (optionnel)
            
        Returns:
            Features normalisées
        """
        print("Normalisation des features...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def encode_target(self, y):
        """
        Encode la variable cible
        
        Args:
            y: Variable cible (koi_disposition)
            
        Returns:
            Variable cible encodée
        """
        # Mapper les valeurs textuelles aux valeurs numériques
        y_encoded = y.map(self.target_mapping)
        
        print(f"Distribution des classes:")
        for label, code in self.target_mapping.items():
            count = (y_encoded == code).sum()
            percentage = (count / len(y_encoded)) * 100
            print(f"   {label} ({code}): {count} ({percentage:.1f}%)")
        
        return y_encoded
    
    def preprocess_for_training(self, filepath, test_size=0.2, random_state=42):
        """
        Pipeline complet de préprocessing pour l'entraînement
        
        Args:
            filepath: Chemin vers le fichier CSV
            test_size: Proportion du jeu de test
            random_state: Seed pour la reproductibilité
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        from sklearn.model_selection import train_test_split
        
        print("DÉBUT DU PREPROCESSING")
        
        # 1. Charger les données
        df = self.load_data(filepath)
        
        # 2. Séparer features et target
        if 'koi_disposition' not in df.columns:
            raise ValueError("La colonne 'koi_disposition' n'existe pas dans le dataset")
        
        y = df['koi_disposition']
        
        # 3. Sélectionner les features
        X, self.feature_names = self.select_features(df)
        
        # 4. Gérer les valeurs manquantes
        X = self.handle_missing_values(X)
        
        # 5. Créer des features engineered
        X = self.create_engineered_features(X)
        
        # 6. Encoder la target
        y = self.encode_target(y)
        
        # 7. Split train/test
        print(f"\nSéparation train/test ({int((1-test_size)*100)}%/{int(test_size*100)}%)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"   Train: {X_train.shape[0]} observations")
        print(f"   Test: {X_test.shape[0]} observations")
        
        # 8. Normaliser
        X_train, X_test = self.normalize_features(X_train, X_test)
        
        print("PREPROCESSING TERMINÉ")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_single_input(self, input_data):
        """
        Préprocesse une seule observation pour la prédiction
        
        Args:
            input_data: Dictionnaire avec les features
            
        Returns:
            Features préprocessées
        """
        # Créer un DataFrame à partir de l'input
        df = pd.DataFrame([input_data])
        
        # Remplir les valeurs manquantes avec les médianes sauvegardées
        for col in df.columns:
            if df[col].isnull().any() and col in self.feature_medians:
                df[col].fillna(self.feature_medians[col], inplace=True)
        
        # Créer des features engineered
        df = self.create_engineered_features(df)
        
        # S'assurer que toutes les colonnes sont présentes et dans le bon ordre
        # Obtenir les colonnes attendues depuis le scaler
        if hasattr(self.scaler, 'feature_names_in_'):
            expected_cols = self.scaler.feature_names_in_
            # Ajouter les colonnes manquantes avec 0
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = 0
            # Réorganiser dans le bon ordre
            df = df[expected_cols]
        
        # Normaliser
        df_scaled = self.scaler.transform(df)
        df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
        
        return df_scaled


def get_feature_importance_names():
    """
    Retourne les noms des features importantes pour l'affichage
    """
    return {
        'koi_period': 'Période Orbitale',
        'koi_duration': 'Durée du Transit',
        'koi_depth': 'Profondeur du Transit',
        'koi_prad': 'Rayon Planétaire',
        'koi_teq': 'Température d\'Équilibre',
        'koi_insol': 'Flux d\'Insolation',
        'koi_steff': 'Température Stellaire',
        'koi_srad': 'Rayon Stellaire',
        'koi_model_snr': 'Signal-to-Noise Ratio',
        'koi_score': 'Score de Disposition',
    }
