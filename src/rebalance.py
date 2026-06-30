# %% [markdown]
# # rebalance — Trade list de rebalanceo con costos
#
# Módulo L del notebook legacy. Genera la trade list desde la posición actual
# hacia los pesos objetivo, con comisión estimada y turnover.
#
# Lógica y fórmulas idénticas al legacy.

# %%
import numpy as np
import pandas as pd

import plotly.graph_objects as go

from theme import apl, C_OK, C_ERR


# %% [markdown]
# ## Cálculo de rebalanceo

# %%
def calcular_rebalanceo(tickers, pesos_objetivo, precios_actuales,
                        nav, comision_pct, posiciones_actuales=None):
    """
    Genera trade list de rebalanceo desde posición actual a pesos objetivo.

    posiciones_actuales: dict {ticker: n_acciones} — si None, asume 0 (posición nueva)
    Retorna DataFrame con operaciones, costo estimado y turnover.
    """
    if posiciones_actuales is None:
        posiciones_actuales = {tk: 0 for tk in tickers}

    trades = []
    for tk, w_obj in zip(tickers, pesos_objetivo):
        precio  = precios_actuales.get(tk, np.nan)
        if np.isnan(precio): continue
        usd_obj = nav * w_obj
        acc_obj = usd_obj / precio
        acc_act = posiciones_actuales.get(tk, 0)
        delta   = acc_obj - acc_act
        usd_trade = abs(delta) * precio
        costo   = usd_trade * comision_pct
        w_act   = (acc_act * precio) / nav if nav > 0 else 0
        trades.append({
            'Ticker':         tk,
            'Peso Actual':    w_act,
            'Peso Objetivo':  w_obj,
            'Gap':            w_obj - w_act,
            'Precio':         precio,
            'Acc. Actuales':  acc_act,
            'Acc. Objetivo':  round(acc_obj, 2),
            'Delta Acciones': round(delta, 2),
            'USD Trade':      round(usd_trade, 2),
            'Dirección':      'COMPRAR' if delta > 0.5 else ('VENDER' if delta < -0.5 else 'MANTENER'),
            'Comisión Est.':  round(costo, 2),
        })

    df = pd.DataFrame(trades).set_index('Ticker')
    df['Turnover Total USD'] = df['USD Trade'].sum()
    df['Costo Total Est.']   = df['Comisión Est.'].sum()
    df['Turnover %']         = df['USD Trade'].sum() / nav
    return df


# %%
def fig_rebalanceo(df_reb, titulo='Rebalanceo hacia PMS'):
    """Gráfica de waterfall: gap de pesos actual vs objetivo."""
    df = df_reb[df_reb['Dirección'] != 'MANTENER'].copy()
    colores = [C_OK if g > 0 else C_ERR for g in df['Gap']]
    fig = go.Figure(go.Bar(
        x=df.index, y=df['Gap'],
        marker_color=colores,
        text=[f'{v:.2%}' for v in df['Gap']], textposition='outside',
        hovertemplate='%{x}<br>Gap: %{y:.2%}<br>USD: $%{customdata:,.0f}',
        customdata=df['USD Trade']
    ))
    fig.add_hline(y=0, line_color='#484f58')
    fig.update_yaxes(title='Diferencia de Peso', tickformat='.0%')
    return apl(fig, titulo, alto=380)
