# %% [markdown]
# # stress — Stress testing histórico
#
# Módulo F del notebook legacy. Simula el comportamiento del portafolio en
# 7 períodos históricos de crisis (2008, 2010, 2013, 2015-16, 2020, 2022, 2023).
#
# Lógica y fórmulas idénticas al legacy.

# %%
import numpy as np
import pandas as pd

import plotly.graph_objects as go

from theme import apl


# %% [markdown]
# ## Períodos de estrés histórico

# %%
PERIODOS_STRESS = {
    'Crisis 2008 (Lehman)':        ('2008-09-01', '2009-03-31'),
    'Flash Crash 2010':            ('2010-04-23', '2010-07-02'),
    'Taper Tantrum 2013':          ('2013-05-22', '2013-06-24'),
    'Caída petróleo 2015-16':      ('2015-08-01', '2016-02-29'),
    'COVID Crash 2020':            ('2020-02-19', '2020-03-23'),
    'Alza de tasas 2022':          ('2022-01-03', '2022-10-14'),
    'Crisis bancaria 2023':        ('2023-03-08', '2023-03-31'),
}


# %% [markdown]
# ## Stress testing

# %%
def stress_testing(precios_diarios, pesos_pmv, pesos_pms, tickers):
    """
    Simula el comportamiento del portafolio en períodos de estrés histórico.
    Usa precios diarios descargados (columnas = tickers).
    """
    resultados = []
    for nombre, (ini, fin) in PERIODOS_STRESS.items():
        sub = precios_diarios.loc[ini:fin, [t for t in tickers if t in precios_diarios.columns]]
        if len(sub) < 5: continue
        rend_sub = np.log(sub / sub.shift(1)).dropna()
        tks = sub.columns.tolist()
        for pesos_arr, port_lbl in [(pesos_pmv, 'PMV'), (pesos_pms, 'PMS')]:
            w = np.array([pesos_arr[tickers.index(t)] if t in tickers else 0 for t in tks])
            if w.sum() > 0: w /= w.sum()
            rp = (rend_sub * w).sum(axis=1)
            retorno_total  = float(np.exp(rp.sum()) - 1)
            max_dd         = float((np.exp(rp.cumsum()) - np.exp(rp.cumsum().cummax())).min())
            vol_anual      = float(rp.std() * np.sqrt(252))
            resultados.append({
                'Período':        nombre,
                'Portafolio':     port_lbl,
                'Retorno Total':  retorno_total,
                'Max Drawdown':   max_dd,
                'Vol Anual':      vol_anual,
                'Días':           len(rp)
            })
    return pd.DataFrame(resultados).set_index(['Período', 'Portafolio'])


# %%
def fig_stress(df_stress):
    """Heatmap de stress testing: retornos por período y portafolio."""
    pivot = df_stress['Retorno Total'].unstack('Portafolio')
    txt   = [[f'{v:.1%}' for v in row] for row in pivot.values]
    fig   = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        text=txt, texttemplate='%{text}', textfont=dict(size=12),
        colorscale='RdYlGn', zmid=0,
        colorbar=dict(title='Retorno', tickformat='.0%', thickness=14,
                      tickfont=dict(color='#c9d1d9'))
    ))
    return apl(fig, 'Stress Testing — Retorno Total por Período de Crisis', alto=380)
