# %% [markdown]
# # regime — Régimen de mercado (K-Means proxy)
#
# Módulo J del notebook legacy. Clasifica el mercado en regímenes
# (Bull / Bear / Lateral) con K-Means sobre retorno y volatilidad rolling.
# Proxy simplificado de un Hidden Markov Model (HMM).
#
# Lógica y fórmulas idénticas al legacy.

# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import plotly.graph_objects as go

from theme import apl, C_BLUE


# %% [markdown]
# ## Detección de regímenes (K-Means)

# %%
def detectar_regimenes(rend_port, n_regimenes=3):
    """
    Clasifica el mercado en regímenes (Bull / Bear / Lateral) usando
    K-Means sobre retorno rolling y volatilidad rolling.
    Proxy simplificado del Hidden Markov Model (HMM).
    """
    r = rend_port.dropna()
    df = pd.DataFrame({
        'ret_21':  r.rolling(21).mean() * 252,
        'vol_21':  r.rolling(21).std() * np.sqrt(252),
        'ret_63':  r.rolling(63).mean() * 252,
    }).dropna()

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    km      = KMeans(n_clusters=n_regimenes, random_state=42, n_init=10)
    labels  = km.fit_predict(X_scaled)
    df['Regimen'] = labels

    # Ordenar regímenes por retorno medio (0=Bear, 1=Lateral, 2=Bull)
    medias = df.groupby('Regimen')['ret_21'].mean().sort_values()
    mapa   = {old: new for new, old in enumerate(medias.index)}
    df['Regimen'] = df['Regimen'].map(mapa)
    nombres = {0: 'Bear 🔴', 1: 'Lateral ⚪', 2: 'Bull 🟢'}
    df['Nombre'] = df['Regimen'].map(nombres)

    # Métricas por régimen
    stats_reg = df.groupby('Nombre').agg(
        Retorno_Anual=('ret_21', 'mean'),
        Volatilidad=('vol_21', 'mean'),
        Dias=('ret_21', 'count')
    ).round(4)

    return df, stats_reg


# %%
def fig_regimenes(precios_port, regimenes_df):
    """Gráfica de precio con fondo coloreado por régimen."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=precios_port.index, y=precios_port,
        line=dict(color=C_BLUE, width=1.5), name='Portafolio'
    ))
    colores_reg = {0: 'rgba(255,123,114,0.12)', 1: 'rgba(139,148,158,0.10)', 2: 'rgba(63,185,80,0.12)'}
    nombres_reg = {0: 'Bear 🔴', 1: 'Lateral ⚪', 2: 'Bull 🟢'}
    prev_reg, prev_date = None, None
    for date, row in regimenes_df.iterrows():
        reg = row['Regimen']
        if prev_reg is None:
            prev_reg, prev_date = reg, date; continue
        if reg != prev_reg or date == regimenes_df.index[-1]:
            fig.add_vrect(
                x0=prev_date, x1=date,
                fillcolor=colores_reg.get(prev_reg, 'rgba(0,0,0,0.05)'),
                line_width=0,
                annotation_text=nombres_reg.get(prev_reg, ''),
                annotation_position='top left',
                annotation_font_size=9
            )
            prev_reg, prev_date = reg, date
    fig.update_yaxes(title='Valor (base 100)')
    fig.update_xaxes(title='Fecha')
    return apl(fig, 'Regímenes de Mercado — Bull / Bear / Lateral', alto=460)
