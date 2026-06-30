# %% [markdown]
# # technical — Análisis técnico (SMA / RSI / MACD / Estocástico)
#
# Módulo M del notebook legacy. Indicadores técnicos, detección de señales
# (Golden/Death Cross, RSI, MACD) y gráficas técnicas.
#
# Lógica y fórmulas idénticas al legacy.

# %%
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from theme import apl, PLOTLY_LAYOUT, COLORES, C_OK, C_ERR, C_BLUE, C_WARN


# %% [markdown]
# ## Indicadores

# %%
def calcular_rsi(c, p=14):
    """RSI de Wilder (suavizado exponencial) sobre la serie de cierres `c`."""
    d = c.diff(); g = d.clip(lower=0).ewm(alpha=1 / p, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / p, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def calcular_macd(c, f=12, s=26, sig=9):
    """MACD: retorna (línea MACD, línea señal, histograma)."""
    m = c.ewm(span=f, adjust=False).mean() - c.ewm(span=s, adjust=False).mean()
    sg = m.ewm(span=sig, adjust=False).mean(); return m, sg, m - sg


def calcular_estocastico(c, k=14, d=3, sm=3):
    """Oscilador estocástico: retorna (%K, %D)."""
    lo = c.rolling(k).min(); hi = c.rolling(k).max()
    pk = ((c - lo) / (hi - lo + 1e-10) * 100).rolling(sm).mean()
    return pk, pk.rolling(d).mean()


# %% [markdown]
# ## Detección de señales

# %%
def detectar_señales(c, rsi, ml, ms):
    """Detecta señales técnicas: Golden/Death Cross, RSI extremos, cruces MACD."""
    out = []
    s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
    if len(s50.dropna()) > 2 and len(s200.dropna()) > 2:
        if s50.iloc[-2] < s200.iloc[-2] and s50.iloc[-1] >= s200.iloc[-1]: out.append('Golden Cross (SMA50>SMA200)')
        elif s50.iloc[-2] > s200.iloc[-2] and s50.iloc[-1] <= s200.iloc[-1]: out.append('Death Cross (SMA50<SMA200)')
    r = rsi.iloc[-1]
    if r >= 70: out.append(f'RSI sobrecomprado: {r:.1f}')
    elif r <= 30: out.append(f'RSI sobrevendido: {r:.1f}')
    if ml.iloc[-2] < ms.iloc[-2] and ml.iloc[-1] >= ms.iloc[-1]: out.append('MACD cruce alcista')
    elif ml.iloc[-2] > ms.iloc[-2] and ml.iloc[-1] <= ms.iloc[-1]: out.append('MACD cruce bajista')
    return out or ['Sin señales destacadas (90 dias)']


# %% [markdown]
# ## Gráficas técnicas

# %%
def fig_tecnica(tk, ind):
    """Gráfica técnica completa de un activo: precio+SMAs, volumen, RSI, MACD, estocástico."""
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True,
                        row_heights=[0.38, 0.14, 0.16, 0.16, 0.16],
                        vertical_spacing=0.025,
                        subplot_titles=[f'{tk} Precio+SMAs', 'Volumen', 'RSI(14)', 'MACD(12,26,9)', 'Estocastico(14,3,3)'])
    idx = ind['close'].index
    fig.add_trace(go.Scatter(x=idx, y=ind['close'], name='Close', line=dict(color=C_BLUE, width=1.8)), row=1, col=1)
    for k, col, nm in [('sma50', '#f0e130', 'SMA50'), ('sma100', C_OK, 'SMA100'), ('sma200', C_ERR, 'SMA200')]:
        fig.add_trace(go.Scatter(x=idx, y=ind[k], name=nm, line=dict(color=col, width=1.2, dash='dot')), row=1, col=1)
    if not ind['volume'].empty:
        vc = ['#3fb950' if d >= 0 else '#ff7b72' for d in ind['close'].diff().fillna(0)]
        fig.add_trace(go.Bar(x=idx, y=ind['volume'], name='Vol', marker_color=vc, opacity=0.7), row=2, col=1)
    fig.add_trace(go.Scatter(x=idx, y=ind['rsi'], name='RSI', line=dict(color='#bc8cff', width=1.5)), row=3, col=1)
    for lv, c in [(70, 'rgba(255,123,114,0.5)'), (30, 'rgba(63,185,80,0.5)')]:
        fig.add_hline(y=lv, line_dash='dash', line_color=c, row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor='rgba(255,123,114,0.06)', row=3, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor='rgba(63,185,80,0.06)', row=3, col=1)
    hc = ['#3fb950' if v >= 0 else '#ff7b72' for v in ind['macd_h'].fillna(0)]
    fig.add_trace(go.Bar(x=idx, y=ind['macd_h'], name='Histo', marker_color=hc, opacity=0.7), row=4, col=1)
    fig.add_trace(go.Scatter(x=idx, y=ind['macd'], name='MACD', line=dict(color=C_BLUE, width=1.5)), row=4, col=1)
    fig.add_trace(go.Scatter(x=idx, y=ind['macd_s'], name='Señal', line=dict(color=C_WARN, width=1.5)), row=4, col=1)
    fig.add_trace(go.Scatter(x=idx, y=ind['pct_k'], name='%K', line=dict(color='#79c0ff', width=1.5)), row=5, col=1)
    fig.add_trace(go.Scatter(x=idx, y=ind['pct_d'], name='%D', line=dict(color=C_WARN, width=1.5, dash='dot')), row=5, col=1)
    for lv in [80, 20]:
        fig.add_hline(y=lv, line_dash='dash', line_color='rgba(255,123,114,0.5)' if lv == 80 else 'rgba(63,185,80,0.5)', row=5, col=1)
    # FIX (port): se excluye también 'legend' del desempaquetado de PLOTLY_LAYOUT.
    # El legacy lo dejaba dentro y además pasaba legend=... explícito, lo que en
    # cualquier versión de Python lanza "multiple values for keyword argument 'legend'".
    fig.update_layout(height=920, **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ('xaxis', 'yaxis', 'margin', 'legend')},
                      margin=dict(l=60, r=30, t=70, b=40), showlegend=True,
                      legend=dict(orientation='h', y=-0.03, font=dict(size=10), bgcolor='rgba(22,27,34,0.9)'))
    for i in range(1, 6):
        fig.update_xaxes(gridcolor='#21262d', linecolor='#30363d', row=i, col=1)
        fig.update_yaxes(gridcolor='#21262d', linecolor='#30363d', row=i, col=1)
    return fig


# %%
def fig_base100(inds, tickers):
    """Rendimiento comparativo de todos los activos en base 100."""
    fig = go.Figure()
    for i, tk in enumerate(tickers):
        c = inds[tk]['close'].dropna(); n = (c / c.iloc[0]) * 100
        fig.add_trace(go.Scatter(x=c.index, y=n, name=tk,
                                 line=dict(color=COLORES[i % len(COLORES)], width=1.8)))
    fig.add_hline(y=100, line_dash='dash', line_color='#484f58')
    fig.update_xaxes(title='Fecha'); fig.update_yaxes(title='Indice base 100')
    return apl(fig, 'Rendimiento Comparativo — Base 100', alto=440)
