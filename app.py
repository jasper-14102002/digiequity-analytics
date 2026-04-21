# ============================================================
# DigiEquity Analytics — Global Digital Economy & Inequality
# Streamlit Application — app.py
# 5 Tabs: Country Explorer | Predictor | World Map |
#         Policy Simulator | DIDI Rankings
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="DigiEquity Analytics",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1a1a2e;
        text-align: center;
        padding: 0.5rem 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #4a4a6a;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.3rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.8rem;
        opacity: 0.9;
    }
    .india-highlight {
        background-color: #f3e5f5;
        border-left: 4px solid #8e44ad;
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    .finding-box {
        background-color: #e8f4fd;
        border-left: 4px solid #3498db;
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING — cached for performance
# ============================================================
@st.cache_data
def load_data():
    df_panel   = pd.read_csv('panel_data_clean.csv')
    df_cluster = pd.read_csv('cluster_data.csv')
    df_topsis  = pd.read_csv('topsis_scores.csv')
    df_divide  = pd.read_csv('step9_digital_divide_trend.csv')
    return df_panel, df_cluster, df_topsis, df_divide

@st.cache_resource
def load_model():
    with open('model_rf.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model_rf_features.json', 'r') as f:
        features = json.load(f)
    return model, features

df_panel, df_cluster, df_topsis, df_divide = load_data()
rf_model, rf_features = load_model()

# Cluster color map
CLUSTER_COLORS = {
    0: '#9b59b6',
    1: '#3498db',
    2: '#f39c12',
    3: '#e74c3c',
    4: '#27ae60'
}

# ============================================================
# HEADER
# ============================================================
st.markdown(
    '<div class="main-header">🌐 DigiEquity Analytics</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">Global Digital Economy & Income '
    'Inequality Explorer | MBA Research Project</div>',
    unsafe_allow_html=True
)
st.markdown("---")

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌍 Country Explorer",
    "🤖 Inequality Predictor",
    "🗺️ World Map",
    "🎯 Policy Simulator",
    "🏆 DIDI Rankings"
])

# ============================================================
# TAB 1: COUNTRY EXPLORER
# ============================================================
with tab1:
    st.subheader("🌍 Country Explorer")
    st.markdown(
        "Explore Gini inequality vs internet penetration "
        "trends for any country (2000–2022)"
    )

    col_sel, col_info = st.columns([1, 2])

    with col_sel:
        countries_available = sorted(
            df_panel['country_name'].unique().tolist()
        )
        selected_country = st.selectbox(
            "Select Country",
            countries_available,
            index=countries_available.index('India')
            if 'India' in countries_available else 0
        )

        # Latest metrics
        country_data = df_panel[
            df_panel['country_name'] == selected_country
        ].sort_values('year')

        latest = country_data.dropna(
            subset=['gini','internet_pct','gdp_per_capita']
        ).iloc[-1] if len(country_data) > 0 else None

        if latest is not None:
            st.markdown("**Latest Available Data:**")
            m1, m2 = st.columns(2)
            with m1:
                st.metric(
                    "Gini Index",
                    f"{latest['gini']:.1f}",
                    help="0=perfect equality, 100=max inequality"
                )
                st.metric(
                    "Internet %",
                    f"{latest['internet_pct']:.1f}%"
                )
            with m2:
                st.metric(
                    "GDP/capita",
                    f"${latest['gdp_per_capita']:,.0f}"
                )
                st.metric(
                    "Electricity",
                    f"{latest['electricity_access']:.1f}%"
                )

            # Cluster info
            cluster_info = df_cluster[
                df_cluster['country_name'] == selected_country
            ]
            if len(cluster_info) > 0:
                cname = cluster_info['cluster_name'].values[0]
                st.markdown(
                    f'<div class="india-highlight">'
                    f'<b>Cluster:</b> {cname}</div>',
                    unsafe_allow_html=True
                )

    with col_info:
        # Dual-axis chart: Gini + Internet
        country_plot = country_data.dropna(
            subset=['gini','internet_pct']
        )

        if len(country_plot) > 0:
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            fig.add_trace(
                go.Scatter(
                    x=country_plot['year'],
                    y=country_plot['gini'],
                    name='Gini Index',
                    line=dict(color='#e74c3c', width=3),
                    mode='lines+markers',
                    marker=dict(size=6)
                ),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(
                    x=country_plot['year'],
                    y=country_plot['internet_pct'],
                    name='Internet %',
                    line=dict(color='#3498db', width=3,
                              dash='dash'),
                    mode='lines+markers',
                    marker=dict(size=6)
                ),
                secondary_y=True
            )

            fig.update_layout(
                title=f"{selected_country} — "
                      f"Gini vs Internet Penetration (2000–2022)",
                hovermode='x unified',
                legend=dict(
                    orientation='h',
                    yanchor='bottom', y=1.02
                ),
                plot_bgcolor='#f8f9fa',
                paper_bgcolor='white',
                height=420
            )
            fig.update_yaxes(
                title_text="Gini Index",
                secondary_y=False,
                color='#e74c3c'
            )
            fig.update_yaxes(
                title_text="Internet Users (%)",
                secondary_y=True,
                color='#3498db'
            )
            fig.update_xaxes(title_text="Year")

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(
                f"Insufficient data for {selected_country}"
            )

    # Digital divide chart below
    st.markdown("---")
    st.subheader("📈 Global Digital Divide Trend (2000–2022)")

    col_h = 'High Income\n(Top 25%)'
    col_m = 'Middle Income'
    col_l = 'Low Income\n(Bottom 25%)'

    # Rename columns if needed
    divide_cols = df_divide.columns.tolist()
    high_col = [c for c in divide_cols if 'High' in c or 'high' in c]
    low_col  = [c for c in divide_cols if 'Low'  in c or 'low'  in c]
    mid_col  = [c for c in divide_cols if 'Mid'  in c or 'mid'  in c]

    fig_divide = go.Figure()

    if high_col:
        fig_divide.add_trace(go.Scatter(
            x=df_divide['year'],
            y=df_divide[high_col[0]],
            name='High Income Countries',
            line=dict(color='#27ae60', width=3),
            fill='tonexty' if low_col else None
        ))
    if mid_col:
        fig_divide.add_trace(go.Scatter(
            x=df_divide['year'],
            y=df_divide[mid_col[0]],
            name='Middle Income Countries',
            line=dict(color='#f39c12', width=2.5,
                      dash='dash')
        ))
    if low_col:
        fig_divide.add_trace(go.Scatter(
            x=df_divide['year'],
            y=df_divide[low_col[0]],
            name='Low Income Countries',
            line=dict(color='#e74c3c', width=3)
        ))

    fig_divide.update_layout(
        title="Digital Divide: Internet Penetration by Income Group",
        xaxis_title="Year",
        yaxis_title="Average Internet Penetration (%)",
        hovermode='x unified',
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='white',
        height=380,
        annotations=[dict(
            x=2011, y=5,
            text="Divide WIDENED by 27pp<br>(2000→2022)",
            showarrow=True, arrowhead=2,
            font=dict(color='red', size=12),
            bgcolor='lightyellow'
        )]
    )
    st.plotly_chart(fig_divide, use_container_width=True)

# ============================================================
# TAB 2: INEQUALITY PREDICTOR
# ============================================================
with tab2:
    st.subheader("🤖 Inequality Predictor")
    st.markdown(
        "Adjust the sliders to simulate a country profile "
        "and predict its Gini inequality index using the "
        "Random Forest model (CV R²=0.75)"
    )

    col_sliders, col_output = st.columns([1, 1])

    with col_sliders:
        st.markdown("**Input Country Profile:**")

        internet_val = st.slider(
            "Internet Penetration (%)",
            0.0, 100.0, 55.9, 0.5,
            help="% of population using internet"
        )
        gdp_val = st.slider(
            "GDP per Capita (USD PPP)",
            1000, 140000, 8594, 500,
            help="GDP per capita in constant 2017 USD"
        )
        urban_val = st.slider(
            "Urban Population (%)",
            10.0, 100.0, 36.0, 0.5
        )
        trade_val = st.slider(
            "Trade % of GDP",
            10.0, 200.0, 45.0, 1.0
        )
        edu_val = st.slider(
            "Education Spending % GDP",
            0.5, 10.0, 4.5, 0.1
        )
        elec_val = st.slider(
            "Electricity Access (%)",
            50.0, 100.0, 99.1, 0.5
        )
        broad_val = st.slider(
            "Broadband per 100 people",
            0.0, 50.0, 2.65, 0.5
        )

    with col_output:
        # Prepare input for RF model
        log_gdp_val    = np.log(max(gdp_val, 1))
        internet_sq_val = internet_val ** 2

        feature_map = {
            'internet_pct':       internet_val,
            'log_gdp':            log_gdp_val,
            'urban_pct':          urban_val,
            'trade_gdp':          trade_val,
            'edu_spend_gdp':      edu_val,
            'electricity_access': elec_val,
            'broadband_per100':   broad_val,
            'internet_sq':        internet_sq_val
        }

        input_values = [
            feature_map.get(f, 0) for f in rf_features
        ]
        X_input = np.array(input_values).reshape(1, -1)
        predicted_gini = rf_model.predict(X_input)[0]

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=predicted_gini,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Predicted Gini Index",
                   'font': {'size': 20}},
            delta={
                'reference': 40,
                'increasing': {'color': '#e74c3c'},
                'decreasing': {'color': '#27ae60'}
            },
            gauge={
                'axis': {'range': [20, 65],
                         'tickwidth': 1},
                'bar': {'color': '#8e44ad', 'thickness': 0.3},
                'steps': [
                    {'range': [20, 30],
                     'color': '#27ae60', 'name': 'Low'},
                    {'range': [30, 40],
                     'color': '#f39c12', 'name': 'Moderate'},
                    {'range': [40, 55],
                     'color': '#e74c3c', 'name': 'High'},
                    {'range': [55, 65],
                     'color': '#c0392b', 'name': 'Very High'}
                ],
                'threshold': {
                    'line': {'color': 'black', 'width': 3},
                    'thickness': 0.75,
                    'value': 40
                }
            }
        ))
        fig_gauge.update_layout(
            height=320,
            paper_bgcolor='white',
            font={'color': '#1a1a2e'}
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Classification
        if predicted_gini > 40:
            st.error(
                f"⚠️ **HIGH INEQUALITY** — Predicted Gini: "
                f"{predicted_gini:.2f}"
            )
        elif predicted_gini > 30:
            st.warning(
                f"🟡 **MODERATE INEQUALITY** — Predicted Gini: "
                f"{predicted_gini:.2f}"
            )
        else:
            st.success(
                f"✅ **LOW INEQUALITY** — Predicted Gini: "
                f"{predicted_gini:.2f}"
            )

        # Feature contribution chart
        st.markdown("**Feature Importance (RF Model):**")
        feat_imp = pd.DataFrame({
            'Feature': ['Electricity', 'Urban %',
                        'Trade GDP', 'Log GDP',
                        'Edu Spend', 'Broadband',
                        'Internet²', 'Internet %'],
            'Importance': [0.449, 0.218, 0.189,
                           0.060, 0.049, 0.026,
                           0.005, 0.004]
        }).sort_values('Importance')

        fig_imp = px.bar(
            feat_imp, x='Importance', y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='RdYlGn',
            title="Feature Importance (MDI)"
        )
        fig_imp.update_layout(
            height=280,
            paper_bgcolor='white',
            plot_bgcolor='#f8f9fa',
            showlegend=False,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# ============================================================
# TAB 3: WORLD MAP
# ============================================================
with tab3:
    st.subheader("🗺️ World Map")

    map_col1, map_col2 = st.columns([1, 3])

    with map_col1:
        map_variable = st.radio(
            "Select Variable to Map:",
            ["Gini Index", "Internet %",
             "Cluster", "DIDI Rank (Equal)"],
            index=0
        )
        st.markdown("---")
        st.markdown("**Map Guide:**")
        if map_variable == "Cluster":
            for k, name in {
                0: "Infrastructure Gap",
                1: "Advanced Digital",
                2: "Developing Digital",
                3: "Unequal Digitalizers"
            }.items():
                color = list(CLUSTER_COLORS.values())[k]
                st.markdown(
                    f'<span style="color:{color}">■</span> '
                    f'C{k}: {name}',
                    unsafe_allow_html=True
                )
        st.markdown("---")
        st.markdown(
            '<div class="india-highlight">'
            '⭐ India highlighted on map</div>',
            unsafe_allow_html=True
        )

    with map_col2:
        # Prepare map data
        # Get country codes
        country_codes = df_panel[[
            'country_name','country_code'
        ]].drop_duplicates()

        # Latest values per country
        map_data = (
            df_panel.dropna(subset=['gini','internet_pct'])
            .sort_values('year')
            .groupby('country_name')
            .last()
            .reset_index()
        )
        map_data = map_data.merge(
            country_codes, on='country_name', how='left'
        )
        map_data = map_data.merge(
            df_cluster[['country_name','cluster',
                         'cluster_name']].drop_duplicates(),
            on='country_name', how='left'
        )
        map_data = map_data.merge(
            df_topsis[['country_name','didi_rank_equal',
                        'didi_score_equal']],
            on='country_name', how='left'
        )
        map_data = map_data.dropna(subset=['country_code'])
        map_data = map_data[map_data['country_code'].str.len() == 3]

        if map_variable == "Gini Index":
            fig_map = px.choropleth(
                map_data,
                locations='country_code',
                color='gini',
                hover_name='country_name',
                hover_data={
                    'country_code': False,
                    'gini': ':.1f',
                    'internet_pct': ':.1f'
                },
                color_continuous_scale='RdYlGn_r',
                range_color=[25, 65],
                title="Gini Index — Higher = More Inequality"
            )
        elif map_variable == "Internet %":
            fig_map = px.choropleth(
                map_data,
                locations='country_code',
                color='internet_pct',
                hover_name='country_name',
                hover_data={
                    'country_code': False,
                    'internet_pct': ':.1f',
                    'gini': ':.1f'
                },
                color_continuous_scale='Blues',
                title="Internet Penetration (%)"
            )
        elif map_variable == "Cluster":
            fig_map = px.choropleth(
                map_data.dropna(subset=['cluster']),
                locations='country_code',
                color='cluster_name',
                hover_name='country_name',
                hover_data={
                    'country_code': False,
                    'gini': ':.1f',
                    'internet_pct': ':.1f'
                },
                color_discrete_sequence=[
                    '#9b59b6','#3498db','#f39c12','#e74c3c'
                ],
                title="K-Means Clusters (K=4)"
            )
        else:
            fig_map = px.choropleth(
                map_data.dropna(subset=['didi_rank_equal']),
                locations='country_code',
                color='didi_rank_equal',
                hover_name='country_name',
                hover_data={
                    'country_code': False,
                    'didi_rank_equal': True,
                    'didi_score_equal': ':.4f'
                },
                color_continuous_scale='RdYlGn_r',
                title="DIDI Rank — Lower = Better"
            )

        fig_map.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='natural earth',
                bgcolor='#f8f9fa'
            ),
            paper_bgcolor='white',
            height=500,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_map, use_container_width=True)

# ============================================================
# TAB 4: POLICY SIMULATOR
# ============================================================
with tab4:
    st.subheader("🎯 Policy Simulator")
    st.markdown(
        "Select a country, adjust its internet penetration, "
        "and see how the RF model projects the new Gini index."
    )

    sim_col1, sim_col2 = st.columns([1, 1])

    with sim_col1:
        sim_country = st.selectbox(
            "Select Country to Simulate",
            sorted(df_panel['country_name'].unique()),
            index=sorted(
                df_panel['country_name'].unique()
            ).index('India')
            if 'India' in df_panel['country_name'].values
            else 0,
            key='sim_country'
        )

        # Get latest data for selected country
        sim_data = (
            df_panel[df_panel['country_name'] == sim_country]
            .dropna(subset=rf_features[:6])
            .sort_values('year')
        )

        if len(sim_data) > 0:
            sim_latest = sim_data.iloc[-1]

            st.markdown(
                f"**Baseline ({int(sim_latest['year'])}):**"
            )
            base_internet = sim_latest.get(
                'internet_pct', 50.0
            )

            st.metric(
                "Current Internet %",
                f"{base_internet:.1f}%"
            )
            st.metric(
                "Current Gini",
                f"{sim_latest['gini']:.1f}"
                if pd.notna(sim_latest['gini'])
                else "N/A"
            )

            st.markdown("---")
            internet_change = st.slider(
                "Increase Internet Penetration by:",
                0.0, 50.0, 10.0, 0.5,
                format="+%.1f%%"
            )

            new_internet = min(
                base_internet + internet_change, 100.0
            )
            st.info(
                f"New internet penetration: **{new_internet:.1f}%**"
            )

    with sim_col2:
        if len(sim_data) > 0:
            # Build feature vector for prediction
            feat_vals = {}
            for feat in rf_features:
                if feat == 'internet_pct':
                    feat_vals[feat] = new_internet
                elif feat == 'internet_sq':
                    feat_vals[feat] = new_internet ** 2
                elif feat == 'log_gdp':
                    feat_vals[feat] = np.log(
                        max(sim_latest.get(
                            'gdp_per_capita', 10000
                        ), 1)
                    )
                else:
                    feat_vals[feat] = sim_latest.get(
                        feat, df_panel[feat].mean()
                    )

            X_sim = np.array(
                [feat_vals[f] for f in rf_features]
            ).reshape(1, -1)
            new_gini = rf_model.predict(X_sim)[0]

            # Baseline prediction
            feat_base = feat_vals.copy()
            feat_base['internet_pct'] = base_internet
            feat_base['internet_sq']  = base_internet ** 2
            X_base = np.array(
                [feat_base[f] for f in rf_features]
            ).reshape(1, -1)
            base_gini = rf_model.predict(X_base)[0]

            gini_change = new_gini - base_gini

            # Results
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric(
                    "Baseline Gini",
                    f"{base_gini:.2f}"
                )
            with res_col2:
                st.metric(
                    "Projected Gini",
                    f"{new_gini:.2f}",
                    delta=f"{gini_change:.2f}",
                    delta_color="inverse"
                )

            # Waterfall chart
            fig_sim = go.Figure(go.Waterfall(
                name="Gini Change",
                orientation="v",
                measure=["absolute","relative","total"],
                x=["Baseline Gini",
                   f"+{internet_change:.0f}% Internet",
                   "Projected Gini"],
                y=[base_gini, gini_change, 0],
                connector={"line": {"color": "#636efa"}},
                decreasing={"marker": {
                    "color": "#27ae60"
                }},
                increasing={"marker": {
                    "color": "#e74c3c"
                }},
                totals={"marker": {
                    "color": "#8e44ad"
                }}
            ))
            fig_sim.update_layout(
                title=f"{sim_country}: Gini Impact of "
                      f"+{internet_change:.0f}% Internet",
                yaxis_title="Gini Index",
                plot_bgcolor='#f8f9fa',
                paper_bgcolor='white',
                height=340
            )
            st.plotly_chart(fig_sim, use_container_width=True)

            # Peer comparison
            st.markdown("**Peer Country Comparison:**")
            sim_cluster = df_cluster[
                df_cluster['country_name'] == sim_country
            ]
            if len(sim_cluster) > 0:
                peer_cluster = sim_cluster['cluster'].values[0]
                peers = df_cluster[
                    df_cluster['cluster'] == peer_cluster
                ]['country_name'].tolist()
                peer_data = (
                    df_panel[
                        df_panel['country_name'].isin(peers)
                    ]
                    .dropna(subset=['gini','internet_pct'])
                    .sort_values('year')
                    .groupby('country_name')
                    .last()
                    .reset_index()
                    [['country_name','internet_pct','gini']]
                    .sort_values('gini')
                )
                peer_data['Highlight'] = peer_data[
                    'country_name'
                ].apply(
                    lambda x: '⭐ ' + x
                    if x == sim_country else x
                )
                st.dataframe(
                    peer_data.rename(columns={
                        'country_name': 'Country',
                        'internet_pct': 'Internet %',
                        'gini': 'Gini'
                    }).set_index('Country').round(2),
                    height=220
                )

# ============================================================
# TAB 5: DIDI RANKINGS
# ============================================================
with tab5:
    st.subheader("🏆 Digital Inclusive Development Index (DIDI)")
    st.markdown(
        "Country rankings using AHP-weighted TOPSIS across "
        "3 priority scenarios. "
        "**Lower rank = better performance.**"
    )

    # Scenario selector
    scenario = st.radio(
        "Select Priority Scenario:",
        ["A: Equal Weights",
         "B: Digital-Growth First",
         "C: Equity First"],
        horizontal=True
    )

    scenario_map = {
        "A: Equal Weights":         "equal",
        "B: Digital-Growth First":  "digital",
        "C: Equity First":          "equity"
    }
    sc = scenario_map[scenario]

    score_col = f'didi_score_{sc}'
    rank_col  = f'didi_rank_{sc}'

    # Scenario weights info
    weight_info = {
        "A: Equal Weights": {
            'Internet': '20%', 'Gini(inv)': '20%',
            'GDP': '20%', 'Education': '20%',
            'Electricity': '20%'
        },
        "B: Digital-Growth First": {
            'Internet': '38%', 'Gini(inv)': '13%',
            'GDP': '25%', 'Education': '10%',
            'Electricity': '14%'
        },
        "C: Equity First": {
            'Internet': '14%', 'Gini(inv)': '42%',
            'GDP': '8%', 'Education': '27%',
            'Electricity': '9%'
        }
    }
    wi = weight_info[scenario]
    w_cols = st.columns(5)
    for col, (k, v) in zip(w_cols, wi.items()):
        col.metric(k, v)

    st.markdown("---")

    # Search
    search = st.text_input(
        "🔍 Search Country", "",
        placeholder="Type country name..."
    )

    # Prepare display table
    display_cols = [
        'country_name', rank_col, score_col,
        'cluster_name', 'internet_pct',
        'gini', 'didi_rank_equal',
        'didi_rank_digital', 'didi_rank_equity'
    ]
    available_cols = [
        c for c in display_cols
        if c in df_topsis.columns
    ]
    df_display = df_topsis[available_cols].copy()
    df_display = df_display.sort_values(rank_col)

    # Filter by search
    if search:
        df_display = df_display[
            df_display['country_name'].str.contains(
                search, case=False, na=False
            )
        ]

    # Highlight India
    def highlight_india(row):
        if row['country_name'] == 'India':
            return ['background-color: #f3e5f5; '
                    'font-weight: bold'] * len(row)
        elif row[rank_col] <= 10:
            return ['background-color: #e8f5e9'] * len(row)
        elif row[rank_col] >= len(df_topsis) - 10:
            return ['background-color: #fce4ec'] * len(row)
        return [''] * len(row)

    col_rename = {
        'country_name':      'Country',
        rank_col:            f'Rank ({sc.upper()})',
        score_col:           'DIDI Score',
        'cluster_name':      'Cluster',
        'internet_pct':      'Internet %',
        'gini':              'Gini',
        'didi_rank_equal':   'Rank A',
        'didi_rank_digital': 'Rank B',
        'didi_rank_equity':  'Rank C'
    }

    df_show = df_display.rename(
        columns={k: v for k, v in col_rename.items()
                 if k in df_display.columns}
    )

    st.dataframe(
        df_show.style.apply(highlight_india, axis=1)
        .format({
            'DIDI Score': '{:.4f}',
            'Internet %': '{:.1f}',
            'Gini':       '{:.1f}'
        }, na_rep='N/A'),
        height=380,
        use_container_width=True
    )

    st.markdown(
        '<div class="india-highlight">'
        '🇮🇳 <b>India highlighted in purple</b> | '
        'Green = Top 10 | Red = Bottom 10</div>',
        unsafe_allow_html=True
    )

    # Top 20 bar chart
    st.markdown("---")
    top20_col, bot20_col = st.columns(2)

    with top20_col:
        top20 = df_topsis.nsmallest(20, rank_col)
        colors_top = [
            '#8e44ad' if c == 'India' else '#27ae60'
            for c in top20['country_name']
        ]
        fig_top = go.Figure(go.Bar(
            x=top20[score_col],
            y=top20['country_name'],
            orientation='h',
            marker_color=colors_top,
            text=top20[score_col].round(4),
            textposition='outside'
        ))
        fig_top.update_layout(
            title=f"Top 20 DIDI — {scenario}",
            xaxis_title="DIDI Score",
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='white',
            height=520,
            yaxis={'autorange': 'reversed'}
        )
        st.plotly_chart(fig_top, use_container_width=True)

    with bot20_col:
        bot20 = df_topsis.nlargest(20, rank_col)
        colors_bot = [
            '#8e44ad' if c == 'India' else '#e74c3c'
            for c in bot20['country_name']
        ]
        fig_bot = go.Figure(go.Bar(
            x=bot20[score_col],
            y=bot20['country_name'],
            orientation='h',
            marker_color=colors_bot,
            text=bot20[score_col].round(4),
            textposition='outside'
        ))
        fig_bot.update_layout(
            title=f"Bottom 20 DIDI — {scenario}",
            xaxis_title="DIDI Score",
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='white',
            height=520,
            yaxis={'autorange': 'reversed'}
        )
        st.plotly_chart(fig_bot, use_container_width=True)

    # India special callout
    india_topsis = df_topsis[
        df_topsis['country_name'] == 'India'
    ]
    if len(india_topsis) > 0:
        st.markdown("---")
        st.markdown(
            '<div class="india-highlight">'
            '<h4>🇮🇳 India DIDI Analysis</h4>'
            f'<b>Scenario A (Equal):</b> '
            f'Rank #{int(india_topsis["didi_rank_equal"].values[0])} '
            f'| Score: '
            f'{india_topsis["didi_score_equal"].values[0]:.4f}<br>'
            f'<b>Scenario B (Digital-first):</b> '
            f'Rank #{int(india_topsis["didi_rank_digital"].values[0])}'
            f'<br>'
            f'<b>Scenario C (Equity-first):</b> '
            f'Rank #{int(india_topsis["didi_rank_equity"].values[0])}'
            f'<br><br>'
            f'<b>Key Insight:</b> India\'s rank swings 50 places '
            f'across scenarios — reflecting its paradox of '
            f'<b>low inequality (Gini=25.5) but moderate digital '
            f'adoption (55.9%)</b>. Under equity criteria, India '
            f'ranks #22 globally — a strong finding for policy.'
            f'</div>',
            unsafe_allow_html=True
        )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:#666; font-size:0.85rem'>
    DigiEquity Analytics | MBA Applied Business Analytics Project<br>
    Federal Bank TSM Centre of Excellence in Banking, 
    Applied Economics & Financial Markets<br>
    Data: World Bank WDI | Models: Random Forest, 
    K-Means, TOPSIS-AHP | 78 Countries | 2000–2022
    </div>
    """,
    unsafe_allow_html=True
)
