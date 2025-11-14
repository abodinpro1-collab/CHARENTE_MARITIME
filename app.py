import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import numpy as np

# ===========================
# CONFIG & INITIALIZATION
# ===========================

st.set_page_config(
    page_title="Nomadia Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Design moderne & dynamique
st.markdown("""
<style>
    * {
        font-family: 'Segoe UI', 'Helvetica', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f8f9fa 0%, #f0f2f5 100%);
    }
    
    /* Headers */
    .header-main {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0066cc 0%, #0052a3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .header-sub {
        font-size: 1.2rem;
        color: #666;
        font-weight: 500;
    }
    
    /* KPI Cards */
    .kpi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 102, 204, 0.08);
        border-left: 4px solid #0066cc;
        transition: all 0.3s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 102, 204, 0.15);
    }
    
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0066cc;
        line-height: 1;
        margin: 0.5rem 0;
    }
    
    .kpi-label {
        font-size: 0.85rem;
        color: #999;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .kpi-delta {
        font-size: 0.9rem;
        margin-top: 0.5rem;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        display: inline-block;
    }
    
    .kpi-delta.positive {
        background: #d4edda;
        color: #155724;
    }
    
    .kpi-delta.negative {
        background: #f8d7da;
        color: #721c24;
    }
    
    /* Alert Boxes */
    .alert-critical {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 4px solid #e53935;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #fff8e1 0%, #ffe0b2 100%);
        border-left: 4px solid #f57f17;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .alert-info {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 4px solid #1976d2;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .alert-success {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 4px solid #388e3c;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Status Badges */
    .badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-urgent {
        background: #ffebee;
        color: #c62828;
    }
    
    .badge-warning {
        background: #fff3e0;
        color: #e65100;
    }
    
    .badge-active {
        background: #e3f2fd;
        color: #01579b;
    }
    
    .badge-resolved {
        background: #e8f5e9;
        color: #1b5e20;
    }
    
    /* Dividers */
    .divider {
        margin: 2rem 0;
        border-top: 2px solid #e0e0e0;
    }
    
    /* Section headers */
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1a1a1a;
        margin: 1.5rem 0 1rem 0;
        padding-left: 0.75rem;
        border-left: 4px solid #0066cc;
    }
    
    /* Metric comparison */
    .metric-comparison {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .metric-icon {
        font-size: 2rem;
    }
</style>
""", unsafe_allow_html=True)

load_dotenv()

# ===========================
# DATA LOADING & CACHE
# ===========================

AIRTABLE_TOKEN = os.getenv('AIRTABLE_TOKEN')
BASE_ID = 'appKqxxTj10MRgBXJ'
TABLE_NAME = 'Signalements'

headers = {
    'Authorization': f'Bearer {AIRTABLE_TOKEN}',
    'Content-Type': 'application/json'
}

@st.cache_data(ttl=300)
def fetch_airtable_data():
    """Récupère les données depuis Airtable avec pagination"""
    url = f'https://api.airtable.com/v0/{BASE_ID}/{TABLE_NAME}'
    all_records = []
    params = {}
    
    try:
        while True:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            all_records.extend(data.get('records', []))
            
            if 'offset' in data:
                params['offset'] = data['offset']
            else:
                break
        return all_records
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur API Airtable: {e}")
        return []

@st.cache_data(ttl=300)
def process_data(records):
    """Transforme les données Airtable en DataFrame structuré"""
    if not records:
        return pd.DataFrame()
    
    data = []
    for record in records:
        fields = record.get('fields', {})
        
        def safe_join(val):
            if isinstance(val, list):
                return ', '.join(val)
            return val or ''
        
        row = {
            'ID': fields.get('ID'),
            'Commune': safe_join(fields.get('Commune recherche')),
            'Intercommunalité': safe_join(fields.get('Intercommunalité')),
            'Arrondissement': safe_join(fields.get('Arrondissement')),
            'Adresse': fields.get('Adresse du stationnement', ''),
            'Date_Debut': fields.get('Date Début de stationnement'),
            'Date_Fin': fields.get('Date fin de stationnement'),
            'Menages': int(fields.get('Nombre de ménages', 0)),
            'Caravanes': int(fields.get('Nombre de caravanes estimées', 0)),
            'Terrain': fields.get('Statut du terrain', ''),
            'Statut_Stationnement': fields.get('Statut du stationnement', ''),
            'Etat_Gestion': fields.get('Etat de gestion du dossier', ''),
            'Situation': fields.get('Situation du voyageur', ''),
            'Gestionnaire': fields.get('Nom du gestionnaire du stationnement', ''),
            'Référent': fields.get('Référent du Groupe', ''),
            'Nb_Interventions': int(fields.get('Nombre d\'interventions', 0)),
            'Delai_1ere_Intervention': fields.get('Délai en jours pour la première intervention'),
            'Duree_Stationnement': fields.get('Durée en jours du stationnement'),
            'Eau': fields.get('Eau'),
            'Electricite': fields.get('Électricité'),
            'Assainissement': fields.get('Assainissement'),
            'Acteurs': fields.get('Acteurs Mobilisés sur la gestion du Dossier', ''),
            'Journal_Interventions': fields.get('Journal interventions', [])
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # 🔧 Conversion dates - IMPORTANT
    for col in ['Date_Debut', 'Date_Fin']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Remplir les dates manquantes avec un placeholder
    if df['Date_Debut'].isna().any():
        df['Date_Debut'].fillna(pd.Timestamp.now(), inplace=True)
    
    return df

def calculate_urgency_score(row):
    """Calcule un score d'urgence (0-100)"""
    score = 0
    
    # Facteur 1: Pas d'intervention (30pts)
    if row['Nb_Interventions'] == 0:
        score += 30
    
    # Facteur 2: Délai d'intervention élevé (30pts)
    if pd.notna(row['Delai_1ere_Intervention']):
        if row['Delai_1ere_Intervention'] > 30:
            score += 30
        elif row['Delai_1ere_Intervention'] > 14:
            score += 15
    
    # Facteur 3: Taille du groupe (20pts)
    score += min(row['Menages'] / 10 * 20, 20)
    
    # Facteur 4: État du dossier (20pts)
    if row['Etat_Gestion'] in ['À traiter', 'Diagnostic en cours']:
        score += 20
    elif row['Etat_Gestion'] == 'Interlocuteur consulté':
        score += 10
    
    return min(score, 100)

# ===========================
# SIDEBAR & FILTERS
# ===========================

def setup_sidebar():
    """Configure les filtres dans la sidebar"""
    with st.sidebar:
        st.markdown("### 🔍 Filtres & Paramètres")
        
        # Refresh button
        if st.button("🔄 Actualiser", use_container_width=True, key="refresh_btn"):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        
        # Filter section
        st.markdown("**📍 Filtrage Géographique**")
        
        # EPCI filter
        epci_list = sorted(df['Intercommunalité'].dropna().unique().tolist())
        selected_epci = st.multiselect(
            "Intercommunalités",
            options=epci_list,
            default=epci_list,
            key="filter_epci"
        )
        
        # Commune filter
        commune_list = sorted(df['Commune'].dropna().unique().tolist())
        selected_communes = st.multiselect(
            "Communes",
            options=commune_list,
            key="filter_communes"
        )
        
        # Arrondissement filter (NEW)
        arrondissement_list = sorted(df['Arrondissement'].dropna().unique().tolist())
        selected_arrondissements = st.multiselect(
            "Arrondissements",
            options=arrondissement_list,
            key="filter_arrondissements"
        )
        
        st.markdown("**📊 Filtrage Opérationnel**")
        
        # Status filter
        statuts = sorted(df['Etat_Gestion'].dropna().unique().tolist())
        selected_status = st.multiselect(
            "État de gestion",
            options=statuts,
            default=statuts,
            key="filter_status"
        )
        
        # Period filter
        st.markdown("**📅 Période**")
        
        # 🔧 FIX: Vérifier que les dates sont valides
        if not df['Date_Debut'].isna().all():
            date_min = pd.Timestamp.now() - pd.Timedelta(days=365)  # Par défaut: 1 an
            date_max = pd.Timestamp.now()
            
            # Utiliser les vraies dates si disponibles
            real_min = df['Date_Debut'].min()
            real_max = df['Date_Debut'].max()
            
            if pd.notna(real_min) and pd.notna(real_max):
                date_min = real_min
                date_max = real_max
            
            date_range = st.date_input(
                "Sélectionner une plage",
                value=(date_min.date(), date_max.date()),
                min_value=date_min.date(),
                max_value=date_max.date(),
                key="filter_dates"
            )
        else:
            st.warning("⚠️ Pas de dates disponibles dans les données")
            date_range = ()
        
        st.divider()
        
        return {
            'epci': selected_epci,
            'communes': selected_communes,
            'arrondissements': selected_arrondissements,
            'status': selected_status,
            'date_range': date_range
        }

def apply_filters(df, filters):
    """Applique les filtres au dataframe"""
    df_filtered = df.copy()
    
    if filters['epci']:
        df_filtered = df_filtered[df_filtered['Intercommunalité'].isin(filters['epci'])]
    
    if filters['communes']:
        df_filtered = df_filtered[df_filtered['Commune'].isin(filters['communes'])]
    
    if filters['arrondissements']:
        df_filtered = df_filtered[df_filtered['Arrondissement'].isin(filters['arrondissements'])]
    
    if filters['status']:
        df_filtered = df_filtered[df_filtered['Etat_Gestion'].isin(filters['status'])]
    
    # 🔧 FIX: Vérifier que date_range n'est pas vide et a 2 éléments
    if len(filters['date_range']) == 2 and filters['date_range'][0] and filters['date_range'][1]:
        df_filtered = df_filtered[
            (df_filtered['Date_Debut'] >= pd.Timestamp(filters['date_range'][0])) &
            (df_filtered['Date_Debut'] <= pd.Timestamp(filters['date_range'][1]))
        ]
    
    return df_filtered

# ===========================
# LOAD DATA
# ===========================

records = fetch_airtable_data()
df = process_data(records)

if df.empty:
    st.error("❌ Aucune donnée disponible. Vérifiez votre token Airtable.")
    st.stop()

# Calculate urgency scores
df['Urgency_Score'] = df.apply(calculate_urgency_score, axis=1)

# Setup filters
filters = setup_sidebar()
df_filtered = apply_filters(df, filters)

# ===========================
# HEADER
# ===========================

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown('<h1 class="header-main">📊 Nomadia</h1>', unsafe_allow_html=True)
    st.markdown('<p class="header-sub">Suivi des stationnements - Gens du voyage</p>', unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style='text-align: right; color: #999; font-size: 0.9rem;'>
        <p>Mise à jour: <b>{datetime.now().strftime("%d/%m/%Y %H:%M")}</b></p>
        <p>{len(df_filtered)} dossier(s) · {len(df)} total</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ===========================
# MAIN DASHBOARD
# ===========================

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Tableau de Bord",
    "📁 Gestion des Dossiers",
    "🗺️ Analyse Territoriale",
    "📋 Interventions",
    "📈 Performance"
])

# ===========================
# TAB 1: EXECUTIVE DASHBOARD
# ===========================

with tab1:
    st.markdown('<h2 class="section-title">🎯 Vue Exécutive</h2>', unsafe_allow_html=True)
    
    # KPIs Row 1: Activité globale
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        nb_signalements = len(df_filtered)
        prev_signalements = len(df_filtered[df_filtered['Date_Debut'] < pd.Timestamp.now() - timedelta(days=30)])
        delta = ((nb_signalements - prev_signalements) / prev_signalements * 100) if prev_signalements > 0 else 0
        
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">📊 Signalements Actifs</div>
            <div class="kpi-value">{nb_signalements}</div>
            <div class="kpi-delta {'positive' if delta >= 0 else 'negative'}">
                {'↑' if delta > 0 else '↓'} {abs(delta):.0f}% (vs 30j)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        menages = int(df_filtered['Menages'].sum())
        menages_prev = int(df_filtered[df_filtered['Date_Debut'] < pd.Timestamp.now() - timedelta(days=30)]['Menages'].sum())
        delta = ((menages - menages_prev) / menages_prev * 100) if menages_prev > 0 else 0
        
        # 🔧 FIX: Éviter division par zéro
        avg_menages = menages / nb_signalements if nb_signalements > 0 else 0
        
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">👥 Ménages Concernés</div>
            <div class="kpi-value">{menages}</div>
            <div class="kpi-delta {'positive' if delta >= 0 else 'negative'}">
                Moyenne: {avg_menages:.1f}/dossier
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        caravanes = int(df_filtered['Caravanes'].sum())
        # 🔧 FIX: Éviter division par zéro
        ratio = caravanes / menages if menages > 0 else 0
        
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">🚐 Caravanes</div>
            <div class="kpi-value">{caravanes}</div>
            <div class="kpi-delta positive">
                Ratio: {ratio:.2f}/ménage
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        urgents = len(df_filtered[df_filtered['Urgency_Score'] > 70])
        # 🔧 FIX: Éviter division par zéro
        pct_urgent = (urgents / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
        
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">🚨 Urgents (Score >70)</div>
            <div class="kpi-value">{urgents}</div>
            <div class="kpi-delta negative">
                {pct_urgent:.0f}% des dossiers
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div style="margin: 1.5rem 0;"></div>', unsafe_allow_html=True)
    
    # KPIs Row 2: Performance
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delai_moyen = df_filtered['Delai_1ere_Intervention'].mean()
        objectif = "✓" if delai_moyen <= 7 else "✗"
        
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">⏱️ Délai 1ère Intervention</div>
            <div class="kpi-value">{delai_moyen:.1f}j</div>
            <div class="kpi-delta {'positive' if delai_moyen <= 7 else 'negative'}">
                {objectif} Objectif 7j
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        nb_sans_intervention = len(df_filtered[df_filtered['Nb_Interventions'] == 0])
        pct = (nb_sans_intervention / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
        
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">❌ Sans Intervention</div>
            <div class="kpi-value">{nb_sans_intervention}</div>
            <div class="kpi-delta negative">
                {pct:.0f}% des dossiers
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        interv_moyen = df_filtered['Nb_Interventions'].mean()
        
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">📞 Interventions Moyennes</div>
            <div class="kpi-value">{interv_moyen:.1f}</div>
            <div class="kpi-delta positive">
                Total: {df_filtered['Nb_Interventions'].sum():.0f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        resolus = len(df_filtered[df_filtered['Etat_Gestion'] == 'Fin du stationnement'])
        taux = (resolus / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
        
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">✅ Dossiers Résolus</div>
            <div class="kpi-value">{taux:.0f}%</div>
            <div class="kpi-delta positive">
                {resolus}/{len(df_filtered)} dossiers
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # === ÉVOLUTION DE LA PRÉSENCE ===
    st.markdown('<h3 class="section-title">📊 Évolution de la Présence des Voyageurs</h3>', unsafe_allow_html=True)
    
    if not df_filtered['Date_Debut'].isna().all():
        # Préparer les données de flux
        flux_data = []
        
        # Nouvelles installations
        for _, row in df_filtered.iterrows():
            if pd.notna(row['Date_Debut']):
                flux_data.append({
                    'date': row['Date_Debut'],
                    'type': 'Installation',
                    'menages': row['Menages'],
                    'caravanes': row['Caravanes']
                })
        
        # Départs
        for _, row in df_filtered.iterrows():
            if pd.notna(row['Date_Fin']):
                flux_data.append({
                    'date': row['Date_Fin'],
                    'type': 'Départ',
                    'menages': -row['Menages'],
                    'caravanes': -row['Caravanes']
                })
        
        if flux_data:
            df_flux = pd.DataFrame(flux_data)
            df_flux['date'] = pd.to_datetime(df_flux['date'])
            df_flux['semaine'] = df_flux['date'].dt.to_period('W').astype(str)
            
            # Agrégation par semaine
            flux_hebdo = df_flux.groupby(['semaine', 'type']).agg({
                'menages': 'sum',
                'caravanes': 'sum'
            }).reset_index()
            
            # Créer un tableau complet avec toutes les semaines
            all_weeks = pd.date_range(
                start=df_flux['date'].min(),
                end=df_flux['date'].max() + pd.Timedelta(days=7),
                freq='W'
            ).to_period('W').astype(str)
            
            # Calcul de la présence cumulée par semaine
            presence_data = []
            present_menages = 0
            present_caravanes = 0
            
            for semaine in sorted(all_weeks):
                installations_menages = flux_hebdo[
                    (flux_hebdo['semaine'] == semaine) & 
                    (flux_hebdo['type'] == 'Installation')
                ]['menages'].sum()
                
                departs_menages = abs(flux_hebdo[
                    (flux_hebdo['semaine'] == semaine) & 
                    (flux_hebdo['type'] == 'Départ')
                ]['menages'].sum())
                
                installations_caravanes = flux_hebdo[
                    (flux_hebdo['semaine'] == semaine) & 
                    (flux_hebdo['type'] == 'Installation')
                ]['caravanes'].sum()
                
                departs_caravanes = abs(flux_hebdo[
                    (flux_hebdo['semaine'] == semaine) & 
                    (flux_hebdo['type'] == 'Départ')
                ]['caravanes'].sum())
                
                present_menages += installations_menages - departs_menages
                present_caravanes += installations_caravanes - departs_caravanes
                
                presence_data.append({
                    'semaine': semaine,
                    'installations_menages': installations_menages,
                    'departs_menages': departs_menages,
                    'present_menages': max(0, present_menages),
                    'installations_caravanes': installations_caravanes,
                    'departs_caravanes': departs_caravanes,
                    'present_caravanes': max(0, present_caravanes)
                })
            
            df_presence = pd.DataFrame(presence_data)
            
            # Graphique principal: Présence de ménages par semaine
            fig_presence = go.Figure()
            
            # Ligne de présence
            fig_presence.add_trace(go.Scatter(
                x=df_presence['semaine'],
                y=df_presence['present_menages'],
                mode='lines+markers',
                name='Ménages présents',
                line=dict(color='#e74c3c', width=3),
                fill='tozeroy',
                fillcolor='rgba(231, 76, 60, 0.15)',
                hovertemplate='<b>Semaine %{x}</b><br>Présents: %{y} ménages<extra></extra>'
            ))
            
            # Barres d'installations
            fig_presence.add_trace(go.Bar(
                x=df_presence['semaine'],
                y=df_presence['installations_menages'],
                name='Arrivées',
                marker_color='#27ae60',
                opacity=0.6,
                hovertemplate='<b>Semaine %{x}</b><br>Nouvelles arrivées: %{y} ménages<extra></extra>'
            ))
            
            # Barres de départs
            fig_presence.add_trace(go.Bar(
                x=df_presence['semaine'],
                y=df_presence['departs_menages'],
                name='Départs',
                marker_color='#3498db',
                opacity=0.6,
                hovertemplate='<b>Semaine %{x}</b><br>Départs: %{y} ménages<extra></extra>'
            ))
            
            fig_presence.update_layout(
                title="Flux de Présence - Ménages",
                xaxis_title="Semaine",
                yaxis_title="Nombre de ménages",
                hovermode='x unified',
                barmode='group',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=400,
                margin=dict(t=40, b=40, l=60, r=20)
            )
            
            st.plotly_chart(fig_presence, use_container_width=True)
        else:
            st.info("ℹ️ Aucune donnée de flux disponible")
    else:
        st.info("ℹ️ Pas de dates disponibles pour afficher l'évolution de présence")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="section-title">📈 Évolution Temporelle</h3>', unsafe_allow_html=True)
        
        df_evol = df_filtered.copy()
        df_evol['Mois'] = df_evol['Date_Debut'].dt.to_period('M').astype(str)
        evol_counts = df_evol.groupby('Mois').size().reset_index(name='Signalements')
        
        fig = px.area(
            evol_counts,
            x='Mois',
            y='Signalements',
            title="Signalements par mois",
            markers=True,
            line_shape='spline'
        )
        fig.update_traces(
            fillcolor='rgba(0, 102, 204, 0.1)',
            line=dict(color='#0066cc', width=3)
        )
        fig.update_layout(
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
            margin=dict(t=40, b=40, l=50, r=20),
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<h3 class="section-title">🎯 Distribution d\'Urgence</h3>', unsafe_allow_html=True)
        
        df_urgency = df_filtered.copy()
        df_urgency['Urgency_Category'] = pd.cut(
            df_urgency['Urgency_Score'],
            bins=[0, 30, 60, 100],
            labels=['Faible', 'Moyen', 'Critique']
        )
        
        urgency_counts = df_urgency['Urgency_Category'].value_counts().sort_index()
        colors = ['#4caf50', '#ff9800', '#f44336']
        
        fig = px.pie(
            values=urgency_counts.values,
            names=urgency_counts.index,
            title="Répartition par urgence",
            color_discrete_sequence=colors,
            hole=0.4
        )
        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Critical alerts
    st.markdown('<h3 class="section-title">⚠️ Alertes Critiques</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    # Alert 1: Dossiers sans intervention
    dossiers_bloqués = df_filtered[
        (df_filtered['Nb_Interventions'] == 0) & 
        (df_filtered['Etat_Gestion'] != 'Fin du stationnement')
    ]
    
    with col1:
        if len(dossiers_bloqués) > 0:
            st.markdown(f"""
            <div class="alert-critical">
                <b>🚨 Dossiers Bloqués</b><br>
                {len(dossiers_bloqués)} dossier(s) sans aucune intervention<br>
                <small>Action requise</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-success">
                <b>✅ Tous les dossiers ont au moins une intervention</b>
            </div>
            """, unsafe_allow_html=True)
    
    # Alert 2: Délais dépassés
    delais_depasses = df_filtered[
        (df_filtered['Delai_1ere_Intervention'] > 30) &
        (df_filtered['Etat_Gestion'] != 'Fin du stationnement')
    ]
    
    with col2:
        if len(delais_depasses) > 0:
            st.markdown(f"""
            <div class="alert-warning">
                <b>⏱️ Délais Dépassés</b><br>
                {len(delais_depasses)} dossier(s) >30 jours sans intervention<br>
                <small>À prioriser</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-success">
                <b>✅ Tous les délais sont respectés</b>
            </div>
            """, unsafe_allow_html=True)
    
    # Alert 3: Gros groupes
    gros_groupes = df_filtered[df_filtered['Menages'] > 20]
    
    with col3:
        if len(gros_groupes) > 0:
            st.markdown(f"""
            <div class="alert-info">
                <b>📍 Gros Groupes Détectés</b><br>
                {len(gros_groupes)} situation(s) avec >20 ménages<br>
                <small>Total: {gros_groupes['Menages'].sum()} ménages</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-success">
                <b>✅ Pas de gros groupes</b>
            </div>
            """, unsafe_allow_html=True)
    
    # 🔧 NEW: Alert 4 - Durées anormales
    st.markdown("")
    col1, col2, col3 = st.columns(3)
    
    durees_valides = df_filtered[df_filtered['Duree_Stationnement'].notna()]['Duree_Stationnement']
    
    with col1:
        # Dossiers très longs (>90j)
        tres_long = len(df_filtered[df_filtered['Duree_Stationnement'] > 90])
        if tres_long > 0:
            st.markdown(f"""
            <div class="alert-warning">
                <b>⏱️ Très Longue Durée (>90j)</b><br>
                {tres_long} dossier(s) depuis >90 jours<br>
                <small>À investiguer prioritairement</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-success">
                <b>✅ Pas de durées excessives</b>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Dossiers rapides (<3j)
        tres_rapides = len(df_filtered[df_filtered['Duree_Stationnement'] < 3])
        if tres_rapides > 0:
            st.markdown(f"""
            <div class="alert-success">
                <b>⚡ Résolutions Rapides (<3j)</b><br>
                {tres_rapides} dossier(s) réglés très vite<br>
                <small>Efficacité détectée</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-info">
                <b>ℹ️ Pas de résolutions ultra-rapides</b>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        # Distribution: court vs long terme
        if len(durees_valides) > 0:
            court = len(df_filtered[df_filtered['Duree_Stationnement'] < 30])
            pct_court = (court / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
            st.markdown(f"""
            <div class="alert-info">
                <b>📊 Distribution Temporelle</b><br>
                {pct_court:.0f}% en <30 jours<br>
                <small>Moyenne: {durees_valides.mean():.0f}j</small>
            </div>
            """, unsafe_allow_html=True)

# ===========================
# TAB 2: DOSSIERS DETAIL
# ===========================

with tab2:
    st.markdown('<h2 class="section-title">📁 Gestion des Dossiers</h2>', unsafe_allow_html=True)
    
    # Top dossiers urgents
    st.markdown('<h3 class="section-title">🔴 Top 10 Dossiers Prioritaires</h3>', unsafe_allow_html=True)
    
    top_dossiers = df_filtered.nlargest(10, 'Urgency_Score')[
        ['ID', 'Commune', 'Menages', 'Caravanes', 'Etat_Gestion', 'Delai_1ere_Intervention', 'Urgency_Score']
    ].copy()
    
    top_dossiers['Urgency_Score'] = top_dossiers['Urgency_Score'].round(1)
    top_dossiers['Status'] = top_dossiers['Etat_Gestion'].apply(
        lambda x: '🟢 Actif' if x not in ['Fin du stationnement'] else '✅ Résolu'
    )
    
    st.dataframe(
        top_dossiers[['ID', 'Commune', 'Menages', 'Etat_Gestion', 'Delai_1ere_Intervention', 'Urgency_Score']],
        use_container_width=True,
        height=400
    )
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Search & filter
    st.markdown('<h3 class="section-title">🔍 Recherche Avancée</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_term = st.text_input("Rechercher un dossier (ID, commune, gestionnaire)...", key="search_dossier")
    
    with col2:
        sort_by = st.selectbox(
            "Trier par",
            ["Urgence (décroissant)", "Date (récent)", "Ménages (grand)", "Délai (long)", "Interventions (élevé)"],
            key="sort_dossier"
        )
    
    # Apply search & sort
    df_search = df_filtered.copy()
    
    if search_term:
        df_search = df_search[
            df_search.astype(str).apply(
                lambda x: search_term.lower() in x.str.lower()
            ).any(axis=1)
        ]
    
    if sort_by == "Urgence (décroissant)":
        df_search = df_search.sort_values('Urgency_Score', ascending=False)
    elif sort_by == "Date (récent)":
        df_search = df_search.sort_values('Date_Debut', ascending=False)
    elif sort_by == "Ménages (grand)":
        df_search = df_search.sort_values('Menages', ascending=False)
    elif sort_by == "Délai (long)":
        df_search = df_search.sort_values('Delai_1ere_Intervention', ascending=False, na_position='last')
    elif sort_by == "Interventions (élevé)":
        df_search = df_search.sort_values('Nb_Interventions', ascending=False)
    
    # Display results
    display_cols = ['ID', 'Commune', 'Intercommunalité', 'Menages', 'Caravanes', 
                    'Etat_Gestion', 'Gestionnaire', 'Delai_1ere_Intervention', 'Urgency_Score']
    
    df_display = df_search[display_cols].copy()
    df_display['Urgency_Score'] = df_display['Urgency_Score'].round(1)
    df_display.columns = ['ID', 'Commune', 'EPCI', 'Ménages', 'Caravanes', 'État', 'Gestionnaire', 'Délai (j)', 'Score']
    
    st.dataframe(df_display, use_container_width=True, height=500)
    
    st.caption(f"📊 {len(df_search)} dossier(s) trouvé(s)")
    
    # Export button
    csv = df_search.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Télécharger les données (CSV)",
        data=csv,
        file_name=f"nomadia_dossiers_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key="download_dossiers"
    )

# ===========================
# TAB 3: ANALYSE TERRITORIALE
# ===========================

with tab3:
    st.markdown('<h2 class="section-title">🗺️ Analyse Territoriale</h2>', unsafe_allow_html=True)
    
    # Stats by EPCI
    st.markdown('<h3 class="section-title">🛵 Vue par Intercommunalité</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        epci_stats = df_filtered.groupby('Intercommunalité').agg({
            'ID': 'count',
            'Menages': 'sum',
            'Caravanes': 'sum',
            'Nb_Interventions': 'mean'
        }).round(1).sort_values('ID', ascending=False).reset_index()
        
        epci_stats.columns = ['EPCI', 'Signalements', 'Ménages', 'Caravanes', 'Interv. Moy']
        
        fig = px.bar(
            epci_stats,
            x='EPCI',
            y='Signalements',
            color='Ménages',
            hover_data=['Caravanes', 'Interv. Moy'],
            title="Activité par Intercommunalité",
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            paper_bgcolor='white',
            plot_bgcolor='white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            epci_stats.sort_values('Ménages', ascending=False),
            x='EPCI',
            y='Ménages',
            color='Ménages',
            title="Ménages concernés par EPCI",
            color_continuous_scale='Reds'
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            paper_bgcolor='white',
            plot_bgcolor='white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div style="margin: 1.5rem 0;"></div>', unsafe_allow_html=True)
    
    # Communes hot-spots
    st.markdown('<h3 class="section-title">🔥 Communes "Hot-Spots"</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        commune_stats = df_filtered.groupby('Commune').agg({
            'ID': 'count',
            'Menages': 'sum',
            'Urgency_Score': 'mean'
        }).sort_values('ID', ascending=False).head(10).reset_index()
        
        commune_stats.columns = ['Commune', 'Dossiers', 'Ménages', 'Urgence Moy']
        
        fig = px.scatter(
            commune_stats,
            x='Dossiers',
            y='Urgence Moy',
            size='Ménages',
            text='Commune',
            title="Communes: Fréquence vs Urgence",
            color='Urgence Moy',
            color_continuous_scale='RdYlGn_r',
            size_max=50
        )
        fig.update_traces(textposition='top center')
        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 🔧 NEW: Statut du terrain (Privé/Public)
        terrain_stats = df_filtered['Terrain'].value_counts().reset_index()
        terrain_stats.columns = ['Terrain', 'Nombre']
        
        if len(terrain_stats) > 0:
            fig_terrain = px.pie(
                terrain_stats,
                values='Nombre',
                names='Terrain',
                title="Répartition par Statut du Terrain",
                color_discrete_sequence=['#3498db', '#e74c3c', '#27ae60', '#f39c12'],
                hole=0.3
            )
            fig_terrain.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                height=400
            )
            st.plotly_chart(fig_terrain, use_container_width=True)
        else:
            st.info("Aucune donnée de terrain disponible")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # EPCI detailed table
    st.markdown('<h3 class="section-title">📊 Tableau Détaillé par EPCI</h3>', unsafe_allow_html=True)
    
    epci_detailed = df_filtered.groupby('Intercommunalité').agg({
        'ID': 'count',
        'Menages': 'sum',
        'Caravanes': 'sum',
        'Nb_Interventions': ['sum', 'mean'],
        'Delai_1ere_Intervention': 'mean',
        'Urgency_Score': 'mean',
        'Etat_Gestion': lambda x: (x == 'Fin du stationnement').sum()
    }).round(1)
    
    epci_detailed.columns = ['Signalements', 'Ménages', 'Caravanes', 'Total Interv.', 'Interv. Moy', 'Délai Moy (j)', 'Urgence Moy', 'Résolus']
    epci_detailed = epci_detailed.sort_values('Signalements', ascending=False)
    
    st.dataframe(epci_detailed, use_container_width=True, height=300)

# ===========================
# TAB 4: INTERVENTIONS
# ===========================

with tab4:
    st.markdown('<h2 class="section-title">📋 Analyse des Interventions</h2>', unsafe_allow_html=True)
    
    # Extract all interventions
    all_interventions = []
    for _, row in df_filtered.iterrows():
        journal = row.get('Journal_Interventions', [])
        if journal and isinstance(journal, list):
            for intervention in journal:
                if intervention:
                    all_interventions.append({
                        'ID': row['ID'],
                        'Commune': row['Commune'],
                        'EPCI': row['Intercommunalité'],
                        'Type': intervention,
                        'Gestionnaire': row['Gestionnaire']
                    })
    
    if all_interventions:
        df_interv = pd.DataFrame(all_interventions)
        
        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">📞 Total Interventions</div>
                <div class="kpi-value">{len(df_interv)}</div>
                <div class="kpi-delta positive">
                    Moy: {len(df_interv)/len(df_filtered):.1f}/dossier
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            dossiers_with_interv = df_interv['ID'].nunique()
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">📁 Dossiers avec Actions</div>
                <div class="kpi-value">{dossiers_with_interv}</div>
                <div class="kpi-delta positive">
                    {dossiers_with_interv/len(df_filtered)*100:.0f}% des dossiers
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            top_type = df_interv['Type'].value_counts().index[0]
            top_count = df_interv['Type'].value_counts().values[0]
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">🎯 Type le + Courant</div>
                <div class="kpi-value" style="font-size: 1.2rem;">{top_type[:20]}</div>
                <div class="kpi-delta positive">
                    {top_count} fois
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            unique_types = df_interv['Type'].nunique()
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">📊 Types Uniques</div>
                <div class="kpi-value">{unique_types}</div>
                <div class="kpi-delta positive">
                    Diversité des actions
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Top intervention types
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="section-title">🔝 Types d\'interventions les plus fréquents</h3>', unsafe_allow_html=True)
            
            top_interventions = df_interv['Type'].value_counts().head(10).reset_index()
            top_interventions.columns = ['Type', 'Fréquence']
            
            fig = px.bar(
                top_interventions,
                y='Type',
                x='Fréquence',
                orientation='h',
                color='Fréquence',
                color_continuous_scale='Blues',
                text='Fréquence'
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                paper_bgcolor='white',
                plot_bgcolor='white',
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<h3 class="section-title">👥 Gestionnaires Mobilisés</h3>', unsafe_allow_html=True)
            
            gestionnaire_counts = df_interv['Gestionnaire'].value_counts().head(8).reset_index()
            gestionnaire_counts.columns = ['Gestionnaire', 'Interventions']
            
            fig = px.bar(
                gestionnaire_counts,
                x='Gestionnaire',
                y='Interventions',
                color='Interventions',
                color_continuous_scale='Greens',
                text='Interventions'
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                xaxis_tickangle=-45,
                paper_bgcolor='white',
                plot_bgcolor='white',
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Detailed table
        st.markdown('<h3 class="section-title">📋 Historique Complet</h3>', unsafe_allow_html=True)
        
        df_interv_display = df_interv[['ID', 'Commune', 'EPCI', 'Type', 'Gestionnaire']].copy()
        df_interv_display.columns = ['ID Dossier', 'Commune', 'EPCI', 'Type d\'Intervention', 'Gestionnaire']
        
        st.dataframe(df_interv_display, use_container_width=True, height=400)
        
        csv_interv = df_interv.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Télécharger les interventions",
            data=csv_interv,
            file_name=f"nomadia_interventions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="download_interventions"
        )
    
    else:
        st.info("ℹ️ Aucune intervention enregistrée pour cette période")

# ===========================
# TAB 5: PERFORMANCE & BENCHMARKING
# ===========================

with tab5:
    st.markdown('<h2 class="section-title">📈 Performance & Benchmarking</h2>', unsafe_allow_html=True)
    
    # Performance indicators
    st.markdown('<h3 class="section-title">⚡ Indicateurs de Performance</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Délai d'intervention histogram
        df_delays = df_filtered[df_filtered['Delai_1ere_Intervention'].notna()].copy()
        
        if len(df_delays) > 0:
            fig = px.histogram(
                df_delays,
                x='Delai_1ere_Intervention',
                nbins=20,
                title="Distribution des délais d'intervention (jours)",
                color_discrete_sequence=['#0066cc'],
                marginal='box'
            )
            fig.add_vline(x=7, line_dash='dash', line_color='green', annotation_text='Objectif 7j')
            fig.add_vline(x=14, line_dash='dash', line_color='orange', annotation_text='Seuil 14j')
            fig.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Intervention count distribution
        fig = px.histogram(
            df_filtered,
            x='Nb_Interventions',
            nbins=15,
            title="Distribution du nombre d'interventions par dossier",
            color_discrete_sequence=['#27ae60'],
            marginal='box'
        )
        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Gestionnaire performance
    st.markdown('<h3 class="section-title">👥 Performance par Gestionnaire</h3>', unsafe_allow_html=True)
    
    gestionnaire_perf = df_filtered.groupby('Gestionnaire').agg({
        'ID': 'count',
        'Delai_1ere_Intervention': 'mean',
        'Nb_Interventions': 'mean',
        'Etat_Gestion': lambda x: (x == 'Fin du stationnement').sum()
    }).round(1).sort_values('ID', ascending=False).reset_index()
    
    gestionnaire_perf.columns = ['Gestionnaire', 'Dossiers', 'Délai Moy (j)', 'Interv. Moy', 'Résolus']
    gestionnaire_perf['Taux Résolution (%)'] = (
        pd.to_numeric(gestionnaire_perf['Résolus'] / gestionnaire_perf['Dossiers'] * 100, errors='coerce')
        .fillna(0)  # 🔧 FIX: Convertir en numeric d'abord, puis remplir NaN
        .round(0)   # Arrondir
        .astype(int)  # Convertir en entier
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(
            gestionnaire_perf.head(10),
            x='Gestionnaire',
            y='Dossiers',
            color='Délai Moy (j)',
            hover_data=['Interv. Moy', 'Taux Résolution (%)'],
            title="Charge de travail par gestionnaire",
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            paper_bgcolor='white',
            plot_bgcolor='white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(
            gestionnaire_perf[['Gestionnaire', 'Dossiers', 'Délai Moy (j)', 'Taux Résolution (%)']].head(10),
            use_container_width=True,
            height=400
        )
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Status distribution
    st.markdown('<h3 class="section-title">📊 État des Dossiers</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        status_counts = df_filtered['Etat_Gestion'].value_counts().reset_index()
        status_counts.columns = ['État', 'Nombre']
        
        fig = px.pie(
            status_counts,
            values='Nombre',
            names='État',
            title="Répartition des états de gestion",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Taux de résolution par EPCI
        epci_resolution = df_filtered.groupby('Intercommunalité').agg({
            'Etat_Gestion': lambda x: (x == 'Fin du stationnement').sum() / len(x) * 100
        }).round(1).sort_values('Etat_Gestion', ascending=False).reset_index()
        
        epci_resolution.columns = ['EPCI', 'Taux Résolution (%)']
        
        fig = px.bar(
            epci_resolution,
            x='EPCI',
            y='Taux Résolution (%)',
            color='Taux Résolution (%)',
            color_continuous_scale='RdYlGn',
            text='Taux Résolution (%)'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            xaxis_tickangle=-45,
            paper_bgcolor='white',
            plot_bgcolor='white',
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # === TEMPORALITÉS AVANCÉES ===
    st.markdown('<h3 class="section-title">⏰ Analyse Avancée des Temporalités</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Corrélation Délai 1ère intervention vs Durée
        df_tempo = df_filtered[
            (df_filtered['Delai_1ere_Intervention'].notna()) & 
            (df_filtered['Duree_Stationnement'].notna())
        ].copy()
        
        if len(df_tempo) > 0:
            fig_corr_delai_duree = px.scatter(
                df_tempo,
                x='Delai_1ere_Intervention',
                y='Duree_Stationnement',
                size='Menages',
                color='Nb_Interventions',
                hover_data=['Commune', 'ID'],
                title="Impact du Délai sur la Durée de Stationnement",
                labels={'Delai_1ere_Intervention': 'Délai 1ère intervention (j)',
                       'Duree_Stationnement': 'Durée stationnement (j)',
                       'Nb_Interventions': 'Interventions'},
                color_continuous_scale='Blues'
            )
            fig_corr_delai_duree.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                height=400
            )
            st.plotly_chart(fig_corr_delai_duree, use_container_width=True)
        else:
            st.info("Données insuffisantes")
    
    with col2:
        # Box plot: Durée par État de gestion
        df_box = df_filtered[df_filtered['Duree_Stationnement'].notna()].copy()
        
        if len(df_box) > 0:
            fig_box = px.box(
                df_box,
                x='Etat_Gestion',
                y='Duree_Stationnement',
                title="Durée de Stationnement par État",
                labels={'Etat_Gestion': 'État de gestion', 'Duree_Stationnement': 'Durée (jours)'},
                color='Etat_Gestion',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_box.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
                height=400,
                xaxis_tickangle=-45,
                showlegend=False
            )
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Données insuffisantes")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Tableau récapitulatif par EPCI
    st.markdown('<h3 class="section-title">📊 Temporalités par EPCI</h3>', unsafe_allow_html=True)
    
    epci_tempo = df_filtered.groupby('Intercommunalité').agg({
        'ID': 'count',
        'Duree_Stationnement': ['mean', 'median', 'min', 'max'],
        'Delai_1ere_Intervention': 'mean'
    }).round(1)
    
    epci_tempo.columns = ['Signalements', 'Durée Moy (j)', 'Durée Médiane (j)', 'Durée Min (j)', 'Durée Max (j)', 'Délai Moy (j)']
    epci_tempo = epci_tempo.sort_values('Durée Moy (j)', ascending=False)
    
    st.dataframe(epci_tempo, use_container_width=True, height=300)
    
    comparison_table = df_filtered.groupby('Intercommunalité').agg({
        'ID': 'count',
        'Menages': 'sum',
        'Caravanes': 'sum',
        'Delai_1ere_Intervention': 'mean',
        'Nb_Interventions': 'mean',
        'Urgency_Score': 'mean'
    }).round(2).sort_values('ID', ascending=False)
    
    comparison_table.columns = ['Signalements', 'Ménages', 'Caravanes', 'Délai Moy (j)', 'Interv. Moy', 'Urgence Moy']
    
    st.dataframe(comparison_table, use_container_width=True, height=300)
    
    # Export comparison
    csv_comparison = comparison_table.to_csv().encode('utf-8')
    st.download_button(
        "📥 Télécharger la comparaison EPCI",
        data=csv_comparison,
        file_name=f"nomadia_comparison_epci_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key="download_comparison"
    )

# ===========================
# FOOTER
# ===========================

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col2:
    st.markdown(f"""
    <div style='text-align: center; color: #999; font-size: 0.85rem; padding: 1rem;'>
        <p><b>Dashboard Nomadia v2.0</b></p>
        <p>Suivi en temps réel · Gens du Voyage</p>
        <p style='font-size: 0.75rem;'>Actualisation: {datetime.now().strftime("%d/%m/%Y %H:%M")} 
        | Données: {len(df_filtered)}/{len(df)}</p>
    </div>
    """, unsafe_allow_html=True)