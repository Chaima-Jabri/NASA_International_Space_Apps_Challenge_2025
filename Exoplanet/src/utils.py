"""
Fonctions utilitaires pour ExoKeplerAI
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import plotly.graph_objects as go
import plotly.express as px


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Affiche la matrice de confusion
    
    Args:
        y_true: Vraies valeurs
        y_pred: Prédictions
        class_names: Noms des classes
        save_path: Chemin de sauvegarde (optionnel)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matrice de Confusion', fontsize=16, fontweight='bold')
    plt.ylabel('Vraie Classe', fontsize=12)
    plt.xlabel('Classe Prédite', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt


def plot_feature_importance(importance_df, top_n=15, save_path=None):
    """
    Affiche l'importance des features
    
    Args:
        importance_df: DataFrame avec l'importance des features
        top_n: Nombre de features à afficher
        save_path: Chemin de sauvegarde (optionnel)
    """
    importance_top = importance_df.head(top_n).copy()
    
    plt.figure(figsize=(12, 8))
    
    # Plot pour chaque modèle
    x = np.arange(len(importance_top))
    width = 0.25
    
    plt.barh(x - width, importance_top['importance_lgb'], width, label='LightGBM', alpha=0.8)
    plt.barh(x, importance_top['importance_cat'], width, label='CatBoost', alpha=0.8)
    plt.barh(x + width, importance_top['importance_xgb'], width, label='XGBoost', alpha=0.8)
    
    plt.yticks(x, importance_top['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top {top_n} Features les Plus Importantes', fontsize=16, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt


def plot_class_distribution(y, class_names, save_path=None):
    """
    Affiche la distribution des classes
    
    Args:
        y: Variable cible
        class_names: Noms des classes
        save_path: Chemin de sauvegarde (optionnel)
    """
    counts = pd.Series(y).value_counts().sort_index()
    
    plt.figure(figsize=(10, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = plt.bar(range(len(counts)), counts.values, color=colors, alpha=0.8)
    
    plt.xticks(range(len(counts)), [class_names[i] for i in counts.index])
    plt.ylabel('Nombre d\'observations', fontsize=12)
    plt.title('Distribution des Classes', fontsize=16, fontweight='bold')
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(y)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt


def plot_roc_curves(y_true, y_proba, class_names, save_path=None):
    """
    Affiche les courbes ROC pour chaque classe
    
    Args:
        y_true: Vraies valeurs
        y_proba: Probabilités prédites
        class_names: Noms des classes
        save_path: Chemin de sauvegarde (optionnel)
    """
    # Binariser les labels
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    n_classes = len(class_names)
    
    plt.figure(figsize=(10, 8))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs', fontsize=12)
    plt.ylabel('Taux de Vrais Positifs', fontsize=12)
    plt.title('Courbes ROC Multi-Classes', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt


def create_prediction_gauge(confidence, prediction):
    """
    Crée une jauge de confiance pour la prédiction
    
    Args:
        confidence: Niveau de confiance (0-1)
        prediction: Classe prédite
        
    Returns:
        Figure Plotly
    """
    colors = {
        'FALSE POSITIVE': '#FF6B6B',
        'CANDIDATE': '#4ECDC4',
        'CONFIRMED': '#45B7D1'
    }
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Confiance: {prediction}", 'font': {'size': 24}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': colors.get(prediction, '#888888')},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#FFE5E5'},
                {'range': [50, 75], 'color': '#E5F5F5'},
                {'range': [75, 100], 'color': '#E5F0FF'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig


def create_probability_bar_chart(probabilities):
    """
    Crée un graphique en barres des probabilités
    
    Args:
        probabilities: Dictionnaire {classe: probabilité}
        
    Returns:
        Figure Plotly
    """
    classes = list(probabilities.keys())
    probs = [probabilities[c] * 100 for c in classes]
    
    colors_map = {
        'FALSE POSITIVE': '#FF6B6B',
        'CANDIDATE': '#4ECDC4',
        'CONFIRMED': '#45B7D1'
    }
    colors = [colors_map.get(c, '#888888') for c in classes]
    
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=probs,
            marker_color=colors,
            text=[f'{p:.1f}%' for p in probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Probabilités par Classe',
        xaxis_title='Classe',
        yaxis_title='Probabilité (%)',
        yaxis_range=[0, 100],
        height=400,
        showlegend=False
    )
    
    return fig


def format_prediction_result(result):
    """
    Formate le résultat de prédiction pour l'affichage
    
    Args:
        result: Dictionnaire de résultat de prédiction
        
    Returns:
        String formaté
    """
    output = f"""
    🎯 **PRÉDICTION: {result['prediction']}**
    
    📊 **Confiance: {result['confidence']*100:.2f}%**
    
    📈 **Probabilités:**
    - FALSE POSITIVE: {result['probabilities']['FALSE POSITIVE']*100:.2f}%
    - CANDIDATE: {result['probabilities']['CANDIDATE']*100:.2f}%
    - CONFIRMED: {result['probabilities']['CONFIRMED']*100:.2f}%
    
    🤖 **Prédictions Individuelles:**
    - LightGBM: {result['individual_predictions']['LightGBM']}
    - CatBoost: {result['individual_predictions']['CatBoost']}
    - XGBoost: {result['individual_predictions']['XGBoost']}
    """
    
    return output


def get_sample_input_data():
    """
    Retourne des données d'exemple pour tester l'interface
    
    Returns:
        Dictionnaire avec des valeurs d'exemple
    """
    return {
        'koi_period': 3.52,
        'koi_duration': 2.48,
        'koi_depth': 615.8,
        'koi_impact': 0.146,
        'koi_prad': 2.26,
        'koi_teq': 1769.0,
        'koi_insol': 141.0,
        'koi_steff': 6117.0,
        'koi_slogg': 4.467,
        'koi_srad': 0.927,
        'koi_model_snr': 35.8,
        'koi_score': 1.0,
        'koi_fpflag_nt': 0,
        'koi_fpflag_ss': 0,
        'koi_fpflag_co': 0,
        'koi_fpflag_ec': 0,
        'koi_kepmag': 11.932
    }


def validate_input_data(data):
    """
    Valide les données d'entrée
    
    Args:
        data: Dictionnaire de données
        
    Returns:
        Tuple (is_valid, error_message)
    """
    required_fields = [
        'koi_period', 'koi_duration', 'koi_depth', 'koi_impact',
        'koi_prad', 'koi_teq', 'koi_insol', 'koi_steff',
        'koi_slogg', 'koi_srad', 'koi_model_snr', 'koi_score',
        'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co',
        'koi_fpflag_ec', 'koi_kepmag'
    ]
    
    # Vérifier les champs manquants
    missing_fields = [f for f in required_fields if f not in data or data[f] is None]
    
    if missing_fields:
        return False, f"Champs manquants: {', '.join(missing_fields)}"
    
    # Vérifier les valeurs numériques
    for field in required_fields:
        try:
            float(data[field])
        except (ValueError, TypeError):
            return False, f"Valeur invalide pour {field}: doit être numérique"
    
    # Vérifier les plages de valeurs
    if data['koi_period'] <= 0:
        return False, "La période orbitale doit être positive"
    
    if data['koi_duration'] <= 0:
        return False, "La durée du transit doit être positive"
    
    if data['koi_prad'] <= 0:
        return False, "Le rayon planétaire doit être positif"
    
    return True, ""


def generate_report(model, X_test, y_test, feature_names, save_dir='reports'):
    """
    Génère un rapport complet d'évaluation
    
    Args:
        model: Modèle entraîné
        X_test: Features de test
        y_test: Target de test
        feature_names: Noms des features
        save_dir: Répertoire de sauvegarde
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n📊 Génération du rapport dans '{save_dir}/'...")
    
    # Prédictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    class_names = ['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED']
    
    # 1. Matrice de confusion
    plot_confusion_matrix(y_test, y_pred, class_names, 
                         save_path=f'{save_dir}/confusion_matrix.png')
    plt.close()
    
    # 2. Importance des features
    importance = model.get_feature_importance(feature_names)
    plot_feature_importance(importance, save_path=f'{save_dir}/feature_importance.png')
    plt.close()
    
    # 3. Distribution des classes
    plot_class_distribution(y_test, class_names, 
                           save_path=f'{save_dir}/class_distribution.png')
    plt.close()
    
    # 4. Courbes ROC
    plot_roc_curves(y_test, y_proba, class_names, 
                   save_path=f'{save_dir}/roc_curves.png')
    plt.close()
    
    print("   ✅ Rapport généré avec succès")
    print(f"   📁 Fichiers sauvegardés dans '{save_dir}/'")
