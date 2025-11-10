"""
Script d'entraînement principal pour ExoKeplerAI
"""

import sys
import os
from src.preprocessing import KeplerDataPreprocessor
from src.model import ExoplanetEnsembleClassifier
from src.utils import generate_report
import joblib


def main():
    """
    Fonction principale d'entraînement
    """
    print(" "*15 + "🌟 EXOKEPLAI - ENTRAÎNEMENT 🌟")

    
    # Configuration
    DATA_PATH = 'Dataset/Kepler.csv'
    MODELS_DIR = 'models'
    REPORTS_DIR = 'reports'
    
    # 1. PREPROCESSING
    print("ÉTAPE 1: PREPROCESSING DES DONNÉES")
    
    preprocessor = KeplerDataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.preprocess_for_training(
        DATA_PATH,
        test_size=0.2,
        random_state=42
    )
    
    # Sauvegarder le preprocessor
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(preprocessor, f'{MODELS_DIR}/preprocessor.pkl')
    print(f"\nPreprocessor sauvegardé dans '{MODELS_DIR}/preprocessor.pkl'")
    
    # 2. ENTRAÎNEMENT
    print("ÉTAPE 2: ENTRAÎNEMENT DES MODÈLES")
    
    model = ExoplanetEnsembleClassifier(random_state=42)
    
    # Créer un ensemble de validation pour l'early stopping
    from sklearn.model_selection import train_test_split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    # Entraîner avec validation
    model.train(X_train_split, y_train_split, X_val, y_val)
    
    # 3. ÉVALUATION
    print("ÉTAPE 3: ÉVALUATION DES PERFORMANCES")
    
    metrics = model.evaluate(X_test, y_test, detailed=True)
    
    # 4. IMPORTANCE DES FEATURES
    print("ÉTAPE 4: ANALYSE DES FEATURES")
    
    feature_importance = model.get_feature_importance(X_train.columns.tolist(), top_n=15)
    print("\nTop 15 Features les Plus Importantes:")
    print(feature_importance[['feature', 'importance_avg']].to_string(index=False))
    
    # 5. SAUVEGARDE
    print("ÉTAPE 5: SAUVEGARDE DES MODÈLES")

    
    model.save_models(MODELS_DIR)
    
    # Sauvegarder les noms de features
    joblib.dump(X_train.columns.tolist(), f'{MODELS_DIR}/feature_names.pkl')
    print(f"   Noms des features sauvegardés")
    
    # 6. GÉNÉRATION DU RAPPORT
    print("ÉTAPE 6: GÉNÉRATION DU RAPPORT")
    
    generate_report(model, X_test, y_test, X_train.columns.tolist(), REPORTS_DIR)
    
    # 7. RÉSUMÉ FINAL
    print("ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print("\nRÉSUMÉ DES PERFORMANCES:")
    print(f"   Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"   Precision: {metrics['precision']*100:.2f}%")
    print(f"   Recall:    {metrics['recall']*100:.2f}%")
    print(f"   F1-Score:  {metrics['f1_score']*100:.2f}%")
    
    print("\n FICHIERS GÉNÉRÉS:")
    print(f"   Modèles: {MODELS_DIR}/")
    print(f"   Rapports: {REPORTS_DIR}/")
    
    print("\n PROCHAINE ÉTAPE:")
    print("   Lancez l'interface web avec: python app.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERREUR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
