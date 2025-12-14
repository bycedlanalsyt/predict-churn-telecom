import streamlit as st
import joblib
import numpy as np
import pandas as pd
import io
from datetime import datetime

# ======================================
# 🌟 CONFIGURATION DE LA PAGE
# ======================================
st.set_page_config(
    page_title="Prédicteur de Churn – Télécom",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================
# 🎨 PALETTE DE COULEURS & CSS PREMIUM
# ======================================
TEAL_PRIMARY = "#0099A4"
TEAL_SECONDARY = "#00C7C7"
TEAL_DARK = "#008B8B"
WHITE = "#FFFFFF"
GRAY_LIGHT = "#F5F7FA"
GRAY_MEDIUM = "#E8ECF1"
SUCCESS_GREEN = "#10B981"
DANGER_RED = "#EF4444"

st.markdown(f"""
    <style>
    /* ===== RESET & BASE ===== */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}
    
    /* ===== HEADER ANIMÉ ===== */
    @keyframes fadeInDown {{
        from {{
            opacity: 0;
            transform: translateY(-20px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    .main-header {{
        background: linear-gradient(135deg, {TEAL_PRIMARY} 0%, {TEAL_SECONDARY} 50%, {TEAL_DARK} 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: {WHITE};
        text-align: center;
        margin-bottom: 2.5rem;
        box-shadow: 0 8px 24px rgba(0, 153, 164, 0.3);
        animation: fadeInDown 0.8s ease-out;
        position: relative;
        overflow: hidden;
    }}
    
    .main-header::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 3s ease-in-out infinite;
    }}
    
    @keyframes pulse {{
        0%, 100% {{ opacity: 0.3; }}
        50% {{ opacity: 0.6; }}
    }}
    
    .main-header h1 {{
        margin: 0;
        font-size: 2.8rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }}
    
    .main-header p {{
        margin: 0.8rem 0 0 0;
        font-size: 1.15rem;
        opacity: 0.95;
        position: relative;
        z-index: 1;
    }}
    
    /* ===== SIDEBAR STYLING ===== */
    .css-1d391kg {{
        background-color: {GRAY_LIGHT};
    }}
    
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {WHITE} 0%, {GRAY_LIGHT} 100%);
    }}
    
    .sidebar-header {{
        background: linear-gradient(135deg, {TEAL_PRIMARY} 0%, {TEAL_SECONDARY} 100%);
        padding: 1.8rem 1.5rem;
        border-radius: 15px;
        color: {WHITE};
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 153, 164, 0.25);
        text-align: center;
    }}
    
    .sidebar-header h2 {{
        margin: 0;
        font-size: 1.4rem;
        font-weight: 600;
    }}
    
    .sidebar-header p {{
        margin: 0.5rem 0 0 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }}
    
    /* ===== CARDS DESIGN ===== */
    .info-card {{
        background: {WHITE};
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border-left: 5px solid {TEAL_PRIMARY};
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    
    .info-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    }}
    
    .info-card h3 {{
        margin: 0 0 1.2rem 0;
        color: {TEAL_DARK};
        font-size: 1.3rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    
    /* ===== PREDICTION CARD ===== */
    @keyframes fadeIn {{
        from {{
            opacity: 0;
            transform: scale(0.95);
        }}
        to {{
            opacity: 1;
            transform: scale(1);
        }}
    }}
    
    .prediction-card {{
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        font-size: 1.2rem;
        margin-top: 1.5rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        animation: fadeIn 0.5s ease-out;
    }}
    
    .prediction-card.churn-risk {{
        background: linear-gradient(135deg, {DANGER_RED} 0%, #DC2626 100%);
        color: {WHITE};
    }}
    
    .prediction-card.no-churn {{
        background: linear-gradient(135deg, {SUCCESS_GREEN} 0%, #059669 100%);
        color: {WHITE};
    }}
    
    .prediction-card h2 {{
        margin: 0 0 1.2rem 0;
        font-size: 2rem;
        font-weight: 700;
    }}
    
    .prediction-card .probability {{
        font-size: 2.5rem;
        font-weight: 700;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }}
    
    .prediction-card .recommendation {{
        font-size: 1rem;
        margin-top: 1.5rem;
        opacity: 0.95;
        line-height: 1.6;
    }}
    
    /* ===== ERROR CARD ===== */
    .error-card {{
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
        border: 2px solid {DANGER_RED};
        padding: 2rem;
        border-radius: 15px;
        margin-top: 1.5rem;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.2);
    }}
    
    .error-card h3 {{
        color: {DANGER_RED};
        margin: 0 0 1rem 0;
        font-size: 1.3rem;
    }}
    
    .error-card p {{
        color: #7F1D1D;
        margin: 0.5rem 0;
        line-height: 1.6;
    }}
    
    /* ===== METRICS STYLING ===== */
    [data-testid="stMetricValue"] {{
        font-size: 1.8rem;
    }}
    
    /* ===== BUTTON STYLING ===== */
    .stButton > button {{
        background: linear-gradient(135deg, {TEAL_PRIMARY} 0%, {TEAL_SECONDARY} 100%);
        color: {WHITE};
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 153, 164, 0.3);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 153, 164, 0.4);
    }}
    
    /* ===== PROGRESS BAR ===== */
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, {TEAL_PRIMARY} 0%, {TEAL_SECONDARY} 100%);
    }}
    
    /* ===== FOOTER ===== */
    .footer {{
        text-align: center;
        color: #6B7280;
        padding: 2rem 1rem;
        margin-top: 3rem;
        border-top: 2px solid {GRAY_MEDIUM};
    }}
    
    .footer p {{
        margin: 0.3rem 0;
    }}
    
    .footer strong {{
        color: {TEAL_DARK};
    }}
    
    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {{
        .main-header h1 {{
            font-size: 2rem;
        }}
        .prediction-card {{
            padding: 1.5rem;
        }}
    }}
    
    /* ===== EXPANDER STYLING ===== */
    .streamlit-expanderHeader {{
        background-color: {WHITE};
        border-radius: 10px;
        padding: 0.8rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    
    .streamlit-expanderContent {{
        background-color: {WHITE};
        border-radius: 10px;
        padding: 1rem;
        margin-top: 0.5rem;
    }}
    </style>
""", unsafe_allow_html=True)

# ======================================
# 🔧 CHARGEMENT DU MODÈLE
# ======================================
@st.cache_resource
def load_model():
    """Charge le modèle XGBoost avec gestion d'erreur"""
    try:
        model = joblib.load("xgb_churn_model.pkl")
        return model, None
    except FileNotFoundError:
        return None, "Le fichier modèle 'xgb_churn_model.pkl' est introuvable."
    except Exception as e:
        return None, f"Erreur lors du chargement du modèle: {str(e)}"

model, model_error = load_model()

if model is None:
    st.markdown("""
        <div class="error-card">
            <h3>❌ Erreur de Chargement</h3>
            <p><strong>Impossible de charger le modèle de prédiction.</strong></p>
            <p>{}</p>
            <p>Veuillez vérifier que le fichier 'xgb_churn_model.pkl' est présent dans le dossier Churn_app.</p>
        </div>
    """.format(model_error), unsafe_allow_html=True)
    st.stop()

# ======================================
# 🔝 HEADER PROFESSIONNEL AVEC ANIMATION
# ======================================
st.markdown("""
    <div class="main-header">
        <h1>📡 Prédicteur de Churn – Télécom</h1>
        <p>
            Modèle Machine Learning : <strong>XGBoost Optimisé (GridSearchCV)</strong>
        </p>
    </div>
""", unsafe_allow_html=True)

# ======================================
# 🧩 SIDEBAR PREMIUM - PARAMÈTRES CLIENT
# ======================================
st.sidebar.markdown("""
    <div class="sidebar-header">
        <h2>⚙️ Paramètres Client</h2>
        <p>Veuillez entrer les informations du client</p>
    </div>
""", unsafe_allow_html=True)

# Section 1: Informations Personnelles
with st.sidebar.expander("👤 Informations Personnelles", expanded=True):
    gender = st.selectbox("Genre", ["Male", "Female"], key="gender", help="Sélectionnez le genre du client")
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1], key="senior", 
                                 help="0 = Non, 1 = Oui (65 ans et plus)")
    Partner = st.selectbox("Partenaire", ["Yes", "No"], key="partner", 
                          help="Le client a-t-il un partenaire ?")
    Dependents = st.selectbox("Enfants à charge", ["Yes", "No"], key="dependents",
                             help="Le client a-t-il des enfants à charge ?")

# Section 2: Services Téléphoniques
with st.sidebar.expander("📞 Services Téléphoniques", expanded=True):
    PhoneService = st.selectbox("Service Téléphonique", ["Yes", "No"], key="phone",
                               help="Le client a-t-il un service téléphonique ?")
    MultipleLines = st.selectbox("Lignes Multiples", ["Yes", "No", "No phone service"], key="multiple",
                                 help="Le client a-t-il plusieurs lignes téléphoniques ?")

# Section 3: Services Internet
with st.sidebar.expander("🌐 Services Internet", expanded=True):
    InternetService = st.selectbox("Type d'Internet", ["DSL", "Fiber optic", "No"], key="internet",
                                  help="Type de service internet")
    OnlineSecurity = st.selectbox("Sécurité en Ligne", ["Yes", "No", "No internet service"], key="security")
    OnlineBackup = st.selectbox("Backup en Ligne", ["Yes", "No", "No internet service"], key="backup")
    DeviceProtection = st.selectbox("Protection Appareil", ["Yes", "No", "No internet service"], key="device")
    TechSupport = st.selectbox("Support Technique", ["Yes", "No", "No internet service"], key="tech")
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], key="streamtv")
    StreamingMovies = st.selectbox("Streaming Films", ["Yes", "No", "No internet service"], key="streammovies")

# Section 4: Contrat & Facturation
with st.sidebar.expander("💼 Contrat & Facturation", expanded=True):
    Contract = st.selectbox("Contrat", ["Month-to-month", "One year", "Two year"], key="contract",
                           help="Type de contrat du client")
    PaperlessBilling = st.selectbox("Facturation électronique", ["Yes", "No"], key="paperless",
                                   help="Facturation sans papier")
    PaymentMethod = st.selectbox("Méthode de Paiement", 
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], key="payment",
        help="Méthode de paiement préférée")

# Section 5: Frais & Ancienneté
with st.sidebar.expander("💰 Frais & Ancienneté", expanded=True):
    tenure = st.number_input("Ancienneté (mois)", min_value=0, max_value=72, value=12, step=1, key="tenure",
                            help="Nombre de mois depuis l'inscription")
    MonthlyCharges = st.number_input("Frais mensuels ($)", min_value=0.0, max_value=200.0, value=70.0, 
                                    step=0.01, key="monthly", format="%.2f",
                                    help="Frais mensuels en dollars")
    TotalCharges = st.number_input("Frais totaux ($)", min_value=0.0, max_value=10000.0, value=300.0, 
                                  step=0.01, key="total", format="%.2f",
                                  help="Total des frais accumulés")

# ======================================
# 📊 ZONE PRINCIPALE - LAYOUT 2 COLONNES
# ======================================
col1, col2 = st.columns([1, 1], gap="large")

# ======================================
# COLONNE GAUCHE: RÉSUMÉ CLIENT
# ======================================
with col1:
    st.markdown("""
        <div class="info-card">
            <h3>📋 Résumé des Informations Client</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Métriques clés en 3 colonnes
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric(
            label="Ancienneté",
            value=f"{tenure} mois",
            delta=f"{tenure/72*100:.0f}% du max" if tenure > 0 else None
        )
    
    with metric_col2:
        st.metric(
            label="Frais Mensuels",
            value=f"${MonthlyCharges:.2f}",
            delta=f"${MonthlyCharges-70:.2f}" if MonthlyCharges != 70 else None
        )
    
    with metric_col3:
        st.metric(
            label="Frais Totaux",
            value=f"${TotalCharges:.2f}",
            delta=f"${TotalCharges-300:.2f}" if TotalCharges != 300 else None
        )
    
    # Détails du contrat
    st.markdown("""
        <div class="info-card">
            <h3>📝 Détails du Contrat</h3>
        </div>
    """, unsafe_allow_html=True)
    
    contract_info = pd.DataFrame({
        "Caractéristique": ["Type de Contrat", "Facturation", "Méthode de Paiement"],
        "Valeur": [
            Contract,
            "Électronique" if PaperlessBilling == "Yes" else "Papier",
            PaymentMethod
        ]
    })
    st.dataframe(contract_info, use_container_width=True, hide_index=True)
    
    # Services actifs
    st.markdown("""
        <div class="info-card">
            <h3>🎯 Services Actifs</h3>
        </div>
    """, unsafe_allow_html=True)
    
    services = []
    if PhoneService == "Yes":
        services.append("📞 Téléphonie")
    if InternetService != "No":
        services.append(f"🌐 Internet ({InternetService})")
    if StreamingTV == "Yes":
        services.append("📺 Streaming TV")
    if StreamingMovies == "Yes":
        services.append("🎬 Streaming Films")
    if OnlineSecurity == "Yes":
        services.append("🔒 Sécurité en ligne")
    if TechSupport == "Yes":
        services.append("🛠️ Support technique")
    
    if services:
        for service in services:
            st.markdown(f"<p style='margin:0.5rem 0; font-size:1rem;'>• {service}</p>", 
                       unsafe_allow_html=True)
    else:
        st.info("ℹ️ Aucun service actif détecté")

# ======================================
# COLONNE DROITE: PRÉDICTION
# ======================================
with col2:
    st.markdown("""
        <div class="info-card">
            <h3>🔮 Prédiction de Churn</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Bouton de prédiction
    predict_button = st.button(
        "🚀 Prévoir le Churn",
        use_container_width=True,
        type="primary"
    )
    
    if predict_button:
        with st.spinner("🔄 Calcul de la prédiction en cours..."):
            # Créer le DataFrame avec les données d'entrée dans le bon ordre
            input_data = pd.DataFrame({
                'gender': [gender],
                'SeniorCitizen': [int(SeniorCitizen)],
                'Partner': [Partner],
                'Dependents': [Dependents],
                'tenure': [int(tenure)],
                'PhoneService': [PhoneService],
                'MultipleLines': [MultipleLines],
                'InternetService': [InternetService],
                'OnlineSecurity': [OnlineSecurity],
                'OnlineBackup': [OnlineBackup],
                'DeviceProtection': [DeviceProtection],
                'TechSupport': [TechSupport],
                'StreamingTV': [StreamingTV],
                'StreamingMovies': [StreamingMovies],
                'Contract': [Contract],
                'PaperlessBilling': [PaperlessBilling],
                'PaymentMethod': [PaymentMethod],
                'MonthlyCharges': [float(MonthlyCharges)],
                'TotalCharges': [float(TotalCharges)]
            })
            
            try:
                # Faire la prédiction
                prediction = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0]
                churn_proba = float(proba[1])  # Conversion en float Python natif
                no_churn_proba = float(proba[0])  # Conversion en float Python natif
                
                # Afficher la barre de progression (corrigée)
                st.progress(churn_proba)
                st.caption(f"Probabilité de churn: {churn_proba:.1%}")
                
                # Affichage du résultat avec carte design
                if prediction == 1:
                    st.markdown(f"""
                        <div class="prediction-card churn-risk">
                            <h2>⚠️ RISQUE DE CHURN ÉLEVÉ</h2>
                            <div class="probability">{churn_proba:.1%}</div>
                            <p class="recommendation">
                                Ce client présente un <strong>risque élevé de résiliation</strong>.<br>
                                <strong>Actions recommandées :</strong> offres promotionnelles ciblées, 
                                contact proactif par le service client, analyse des raisons de mécontentement.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Graphique de probabilités
                    prob_df = pd.DataFrame({
                        'Risque de Churn': [churn_proba],
                        'Client Fidèle': [no_churn_proba]
                    })
                    # Liste de couleurs dans l'ordre des colonnes : ['Risque de Churn', 'Client Fidèle']
                    st.bar_chart(prob_df, use_container_width=True, 
                               color=[DANGER_RED, SUCCESS_GREEN])
                    
                else:
                    st.markdown(f"""
                        <div class="prediction-card no-churn">
                            <h2>✅ CLIENT FIDÈLE</h2>
                            <div class="probability">{churn_proba:.1%}</div>
                            <p class="recommendation">
                                Ce client présente un <strong>faible risque de résiliation</strong>.<br>
                                <strong>Recommandation :</strong> maintenir une relation de qualité, 
                                proposer des services complémentaires, fidélisation continue.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Graphique de probabilités
                    prob_df = pd.DataFrame({
                        'Client Fidèle': [no_churn_proba],
                        'Risque de Churn': [churn_proba]
                    })
                    # Liste de couleurs dans l'ordre des colonnes : ['Client Fidèle', 'Risque de Churn']
                    st.bar_chart(prob_df, use_container_width=True,
                               color=[SUCCESS_GREEN, DANGER_RED])
                
                # Métriques détaillées
                st.markdown("#### 📈 Détails de la Prédiction")
                metric_col1, metric_col2 = st.columns(2)
                
                with metric_col1:
                    delta_value = (churn_proba - 0.5) * 100
                    st.metric(
                        "Probabilité de Churn",
                        f"{churn_proba:.2%}",
                        delta=f"{delta_value:+.1f}% vs seuil 50%",
                        delta_color="inverse" if churn_proba > 0.5 else "normal"
                    )
                
                with metric_col2:
                    st.metric(
                        "Probabilité de Fidélité",
                        f"{no_churn_proba:.2%}",
                        delta=f"{(no_churn_proba - 0.5) * 100:+.1f}% vs seuil 50%"
                    )
                
                # Bouton de téléchargement du rapport
                st.markdown("---")
                
                # Créer un rapport CSV
                report_data = {
                    'Date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                    'Genre': [gender],
                    'Senior Citizen': [SeniorCitizen],
                    'Ancienneté (mois)': [tenure],
                    'Frais Mensuels': [MonthlyCharges],
                    'Frais Totaux': [TotalCharges],
                    'Type de Contrat': [Contract],
                    'Prédiction': ['Churn' if prediction == 1 else 'No Churn'],
                    'Probabilité Churn': [f"{churn_proba:.4%}"],
                    'Probabilité Fidélité': [f"{no_churn_proba:.4%}"]
                }
                report_df = pd.DataFrame(report_data)
                
                csv_buffer = io.StringIO()
                report_df.to_csv(csv_buffer, index=False)
                csv_string = csv_buffer.getvalue()
                
                st.download_button(
                    label="📥 Télécharger le Rapport (CSV)",
                    data=csv_string,
                    file_name=f"churn_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            except Exception as e:
                # Message d'erreur élégant
                error_message = str(e)
                st.markdown(f"""
                    <div class="error-card">
                        <h3>❌ Erreur lors de la Prédiction</h3>
                        <p><strong>Une erreur s'est produite lors du calcul de la prédiction.</strong></p>
                        <p><strong>Détails :</strong> {error_message}</p>
                        <p><strong>Veuillez vérifier :</strong></p>
                        <ul style="margin:0.5rem 0; padding-left:1.5rem; color:#7F1D1D;">
                            <li>Que tous les champs sont correctement remplis</li>
                            <li>Que les valeurs numériques sont valides</li>
                            <li>Que le modèle est correctement chargé</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

# ======================================
# 🦶 FOOTER PROFESSIONNEL
# ======================================
st.markdown("""
    <div class="footer">
        <p><strong>Made by Cédric BOIMIN</strong> — Data & IA</p>
        <p style="font-size:0.85rem; opacity:0.8;">
            Modèle XGBoost optimisé avec GridSearchCV | 
            Application Streamlit Premium
        </p>
    </div>
""", unsafe_allow_html=True)
