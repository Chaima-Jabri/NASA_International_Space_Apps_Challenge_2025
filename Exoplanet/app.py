"""
Interface web Gradio pour ExoKeplerAI
"""

import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os
from src.preprocessing import KeplerDataPreprocessor
from src.model import ExoplanetEnsembleClassifier
from src.utils import (
    create_prediction_gauge, 
    create_probability_bar_chart,
    format_prediction_result,
    get_sample_input_data
)


# Charger les modèles et le preprocessor
MODELS_DIR = 'models'

print("Chargement des modèles...")

try:
    preprocessor = joblib.load(f'{MODELS_DIR}/preprocessor.pkl')
    model = ExoplanetEnsembleClassifier()
    model.load_models(MODELS_DIR)
    feature_names = joblib.load(f'{MODELS_DIR}/feature_names.pkl')
    print("Modèles chargés avec succès!")
except Exception as e:
    print(f"Erreur lors du chargement des modèles: {e}")
    print("Veuillez d'abord entraîner les modèles avec: python train_model.py")
    preprocessor = None
    model = None
    feature_names = None


def predict_exoplanet(
    koi_period, koi_duration, koi_depth, koi_impact,
    koi_prad, koi_teq, koi_insol, koi_steff,
    koi_slogg, koi_srad, koi_model_snr, koi_score,
    koi_fpflag_nt, koi_fpflag_ss, koi_fpflag_co, koi_fpflag_ec,
    koi_kepmag
):
    """
    Fonction de prédiction pour l'interface Gradio
    """
    if model is None or preprocessor is None:
        return (
            "Modèles non chargés. Veuillez entraîner les modèles d'abord.",
            None,
            None
        )
    
    try:
        # Créer le dictionnaire d'input
        input_data = {
            'koi_period': float(koi_period),
            'koi_duration': float(koi_duration),
            'koi_depth': float(koi_depth),
            'koi_impact': float(koi_impact),
            'koi_prad': float(koi_prad),
            'koi_teq': float(koi_teq),
            'koi_insol': float(koi_insol),
            'koi_steff': float(koi_steff),
            'koi_slogg': float(koi_slogg),
            'koi_srad': float(koi_srad),
            'koi_model_snr': float(koi_model_snr),
            'koi_score': float(koi_score),
            'koi_fpflag_nt': int(koi_fpflag_nt),
            'koi_fpflag_ss': int(koi_fpflag_ss),
            'koi_fpflag_co': int(koi_fpflag_co),
            'koi_fpflag_ec': int(koi_fpflag_ec),
            'koi_kepmag': float(koi_kepmag)
        }
        
        # Préprocesser
        X = preprocessor.preprocess_single_input(input_data)
        
        # Prédire
        result = model.predict_single(X)
        
        # Formater le résultat
        result_text = format_prediction_result(result)
        
        # Créer les visualisations
        gauge_fig = create_prediction_gauge(result['confidence'], result['prediction'])
        bar_fig = create_probability_bar_chart(result['probabilities'])
        
        return result_text, gauge_fig, bar_fig
        
    except Exception as e:
        return f"Erreur lors de la prédiction: {str(e)}", None, None


def predict_from_csv(file):
    """
    Prédiction à partir d'un fichier CSV
    """
    if model is None or preprocessor is None:
        return "Modèles non chargés. Veuillez entraîner les modèles d'abord.", None
    
    try:
        # Lire le CSV avec gestion des permissions Windows
        if file is None:
            return "Aucun fichier uploadé.", None
        
        # Lire directement depuis le chemin temporaire
        import time
        time.sleep(0.1)  # Petit délai pour éviter les conflits de fichiers
        df = pd.read_csv(file.name)
        
        # Vérifier les colonnes requises
        required_cols = [
            'koi_period', 'koi_duration', 'koi_depth', 'koi_impact',
            'koi_prad', 'koi_teq', 'koi_insol', 'koi_steff',
            'koi_slogg', 'koi_srad', 'koi_model_snr', 'koi_score',
            'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co',
            'koi_fpflag_ec', 'koi_kepmag'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return f"Colonnes manquantes dans le CSV: {', '.join(missing_cols)}"
        
        # Préprocesser et prédire pour chaque ligne
        predictions = []
        for idx, row in df.iterrows():
            input_data = row[required_cols].to_dict()
            X = preprocessor.preprocess_single_input(input_data)
            result = model.predict_single(X)
            
            predictions.append({
                'Index': idx,
                'Prédiction': result['prediction'],
                'Confiance': f"{result['confidence']*100:.2f}%",
                'Prob_FALSE_POSITIVE': f"{result['probabilities']['FALSE POSITIVE']*100:.2f}%",
                'Prob_CANDIDATE': f"{result['probabilities']['CANDIDATE']*100:.2f}%",
                'Prob_CONFIRMED': f"{result['probabilities']['CONFIRMED']*100:.2f}%"
            })
        
        # Créer un DataFrame de résultats
        results_df = pd.DataFrame(predictions)
        
        # Statistiques
        stats = f"""
         **RÉSULTATS DE L'ANALYSE**
        
        Total d'observations: {len(results_df)}
        
        **Distribution des prédictions:**
        - FALSE POSITIVE: {(results_df['Prédiction'] == 'FALSE POSITIVE').sum()}
        - CANDIDATE: {(results_df['Prédiction'] == 'CANDIDATE').sum()}
        - CONFIRMED: {(results_df['Prédiction'] == 'CONFIRMED').sum()}
        """
        
        return stats, results_df
        
    except Exception as e:
        return f" Erreur lors du traitement du CSV: {str(e)}", None


def load_example():
    """
    Charge des données d'exemple
    """
    example = get_sample_input_data()
    return [example[key] for key in [
        'koi_period', 'koi_duration', 'koi_depth', 'koi_impact',
        'koi_prad', 'koi_teq', 'koi_insol', 'koi_steff',
        'koi_slogg', 'koi_srad', 'koi_model_snr', 'koi_score',
        'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
        'koi_kepmag'
    ]]


# Créer l'interface Gradio
with gr.Blocks(title="ExoKeplerAI", theme=gr.themes.Soft()) as app:
    
    gr.Markdown("""
    # 🌟 ExoKeplerAI - Identification d'Exoplanètes
    
    ### Système d'IA pour l'analyse des données Kepler
    
    Utilise un modèle d'ensemble learning combinant **LightGBM**, **CatBoost** et **XGBoost** 
    pour identifier les exoplanètes avec une précision optimale.
    
    ---
    """)
    
    with gr.Tabs():
        
        # TAB 1: Prédiction Simple
        with gr.Tab("🔍 Prédiction Simple"):
            gr.Markdown("### Entrez les paramètres de l'objet céleste")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🌍 Paramètres Orbitaux")
                    koi_period = gr.Number(label="Période Orbitale (jours)", value=3.52)
                    koi_duration = gr.Number(label="Durée du Transit (heures)", value=2.48)
                    koi_depth = gr.Number(label="Profondeur du Transit (ppm)", value=615.8)
                    koi_impact = gr.Number(label="Paramètre d'Impact", value=0.146)
                    
                    gr.Markdown("#### 🪐 Paramètres Planétaires")
                    koi_prad = gr.Number(label="Rayon Planétaire (rayons terrestres)", value=2.26)
                    koi_teq = gr.Number(label="Température d'Équilibre (K)", value=1769.0)
                    koi_insol = gr.Number(label="Flux d'Insolation (flux terrestre)", value=141.0)
                    
                with gr.Column():
                    gr.Markdown("#### ⭐ Paramètres Stellaires")
                    koi_steff = gr.Number(label="Température Stellaire (K)", value=6117.0)
                    koi_slogg = gr.Number(label="Gravité de Surface (log10)", value=4.467)
                    koi_srad = gr.Number(label="Rayon Stellaire (rayons solaires)", value=0.927)
                    koi_kepmag = gr.Number(label="Magnitude Kepler", value=11.932)
                    
                    gr.Markdown("#### 📊 Métriques de Qualité")
                    koi_model_snr = gr.Number(label="Signal-to-Noise Ratio", value=35.8)
                    koi_score = gr.Number(label="Score de Disposition", value=1.0)
                    
                with gr.Column():
                    gr.Markdown("#### 🚩 Flags de Faux Positifs")
                    koi_fpflag_nt = gr.Number(label="Not Transit-Like (0 ou 1)", value=0)
                    koi_fpflag_ss = gr.Number(label="Stellar Eclipse (0 ou 1)", value=0)
                    koi_fpflag_co = gr.Number(label="Centroid Offset (0 ou 1)", value=0)
                    koi_fpflag_ec = gr.Number(label="Ephemeris Match (0 ou 1)", value=0)
            
            with gr.Row():
                predict_btn = gr.Button("🚀 Prédire", variant="primary", size="lg")
                example_btn = gr.Button("📝 Charger un Exemple", size="lg")
            
            with gr.Row():
                with gr.Column():
                    result_text = gr.Textbox(label="Résultat de la Prédiction", lines=15)
                with gr.Column():
                    gauge_plot = gr.Plot(label="Jauge de Confiance")
                    bar_plot = gr.Plot(label="Probabilités par Classe")
            
            # Actions
            predict_btn.click(
                fn=predict_exoplanet,
                inputs=[
                    koi_period, koi_duration, koi_depth, koi_impact,
                    koi_prad, koi_teq, koi_insol, koi_steff,
                    koi_slogg, koi_srad, koi_model_snr, koi_score,
                    koi_fpflag_nt, koi_fpflag_ss, koi_fpflag_co, koi_fpflag_ec,
                    koi_kepmag
                ],
                outputs=[result_text, gauge_plot, bar_plot]
            )
            
            example_btn.click(
                fn=load_example,
                inputs=[],
                outputs=[
                    koi_period, koi_duration, koi_depth, koi_impact,
                    koi_prad, koi_teq, koi_insol, koi_steff,
                    koi_slogg, koi_srad, koi_model_snr, koi_score,
                    koi_fpflag_nt, koi_fpflag_ss, koi_fpflag_co, koi_fpflag_ec,
                    koi_kepmag
                ]
            )
        
        # TAB 2: Prédiction par Lot (CSV)
        with gr.Tab("📊 Prédiction par Lot (CSV)"):
            gr.Markdown("""
            ### Analysez plusieurs observations en une fois
            
            Uploadez un fichier CSV contenant les colonnes suivantes:
            `koi_period`, `koi_duration`, `koi_depth`, `koi_impact`, `koi_prad`, `koi_teq`, 
            `koi_insol`, `koi_steff`, `koi_slogg`, `koi_srad`, `koi_model_snr`, `koi_score`,
            `koi_fpflag_nt`, `koi_fpflag_ss`, `koi_fpflag_co`, `koi_fpflag_ec`, `koi_kepmag`
            """)
            
            csv_file = gr.File(label="Fichier CSV", file_types=[".csv"])
            csv_predict_btn = gr.Button("🚀 Analyser le CSV", variant="primary", size="lg")
            
            csv_stats = gr.Textbox(label="Statistiques", lines=10)
            csv_results = gr.Dataframe(label="Résultats Détaillés")
            
            csv_predict_btn.click(
                fn=predict_from_csv,
                inputs=[csv_file],
                outputs=[csv_stats, csv_results]
            )
        
        # TAB 3: À propos
        with gr.Tab("ℹ️ À propos"):
            gr.Markdown("""
            # ExoKeplerAI
            
            ## 🎯 Objectif
            
            ExoKeplerAI est un système d'intelligence artificielle développé pour analyser 
            automatiquement les données du satellite Kepler et identifier de nouvelles exoplanètes.
            
            ## 🤖 Technologie
            
            Le système utilise un **modèle d'ensemble learning** combinant trois algorithmes 
            de machine learning de pointe:
            
            - **LightGBM**: Gradient boosting rapide et efficace
            - **CatBoost**: Gestion native des variables catégorielles
            - **XGBoost**: Robustesse et précision élevée
            
            ## 📊 Dataset
            
            - **Source**: NASA Kepler Mission
            - **Observations**: 9,564
            - **Features**: 49 caractéristiques
            - **Classes**: 
              - CONFIRMED (2,746 exoplanètes confirmées)
              - CANDIDATE (1,979 candidats)
              - FALSE POSITIVE (4,839 faux positifs)
            
            ## 📈 Performance
            
            Le modèle atteint des performances élevées:
            - Accuracy > 90%
            - Precision > 88%
            - Recall > 85%
            - F1-Score > 87%
            
            ## 👥 Développé pour
            
            **NASA Space Apps Challenge 2025**
            
            ## 📝 Licence
            
            MIT License
            
            ---
            
            **Fait avec ❤️ pour la découverte d'exoplanètes**
            """)
    
    gr.Markdown("""
    ---
    ### 💡 Conseils d'utilisation
    
    - Utilisez le bouton **"Charger un Exemple"** pour voir des valeurs typiques
    - Les **flags de faux positifs** doivent être 0 ou 1
    - Pour l'analyse par lot, assurez-vous que votre CSV contient toutes les colonnes requises
    - La **confiance** indique la certitude du modèle dans sa prédiction
    """)


# Lancer l'application
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
