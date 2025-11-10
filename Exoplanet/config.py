"""
Configuration pour ExoKeplerAI
"""

# Chemins
DATA_PATH = 'Dataset/Kepler.csv'
MODELS_DIR = 'models'
REPORTS_DIR = 'reports'

# Paramètres de preprocessing
PREPROCESSING_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'validation_size': 0.15
}

# Hyperparamètres LightGBM
LIGHTGBM_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 8,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbose': -1,
    'class_weight': 'balanced'
}

# Hyperparamètres CatBoost
CATBOOST_PARAMS = {
    'iterations': 500,
    'learning_rate': 0.05,
    'depth': 8,
    'l2_leaf_reg': 3,
    'random_state': 42,
    'verbose': False,
    'auto_class_weights': 'Balanced'
}

# Hyperparamètres XGBoost
XGBOOST_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 8,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbosity': 0,
    'eval_metric': 'mlogloss'
}

# Poids de l'ensemble
ENSEMBLE_WEIGHTS = {
    'lgb': 0.33,
    'cat': 0.34,
    'xgb': 0.33
}

# Configuration Gradio
GRADIO_CONFIG = {
    'server_name': '0.0.0.0',
    'server_port': 7860,
    'share': False,
    'show_error': True
}

# Mapping des classes
CLASS_MAPPING = {
    'CONFIRMED': 2,
    'CANDIDATE': 1,
    'FALSE POSITIVE': 0
}

CLASS_NAMES = ['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED']

# Features importantes (dans l'ordre de priorité)
IMPORTANT_FEATURES = [
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
