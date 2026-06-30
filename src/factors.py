# %% [markdown]
# # factors — Análisis de factores Fama-French (proxy ETF)
#
# Módulo I del notebook legacy. Construye factores Fama-French proxy desde
# ETFs públicos y corre una regresión OLS del portafolio contra ellos.
#
# Lógica y fórmulas idénticas al legacy. Usa statsmodels para el OLS.

# %%
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from tqdm.auto import tqdm

import plotly.graph_objects as go

from theme import apl, C_ERR, C_GOLD


# %% [markdown]
# ## Descarga de factores Fama-French proxy

# %%
def descargar_factores_ff(inicio, fin):
    """
    Construye factores Fama-French proxy desde ETFs públicos:
    Mkt-Rf: SPY-RF | SMB: IWM-SPY | HML: IVE-IWF | MOM: MTUM-USMV
    """
    etfs = ['SPY', 'IWM', 'IVE', 'IWF', 'MTUM', 'USMV', 'BIL']
    raw = {}
    for tk in tqdm(etfs, desc='Factores FF'):
        try:
            d = yf.download(tk, start=inicio, end=fin, interval='1d',
                            progress=False, auto_adjust=False)
            if not d.empty: raw[tk] = np.log(d['Close'].squeeze() / d['Close'].squeeze().shift(1))
        except: pass
    if 'SPY' not in raw: return None
    rf_d = raw.get('BIL', pd.Series(0, index=raw['SPY'].index))
    mkt  = raw['SPY'] - rf_d
    smb  = raw.get('IWM', raw['SPY']) - raw['SPY']   # Small minus Big
    hml  = raw.get('IVE', raw['SPY']) - raw.get('IWF', raw['SPY'])  # Value minus Growth
    mom  = raw.get('MTUM', raw['SPY']) - raw.get('USMV', raw['SPY'])  # Momentum
    df   = pd.DataFrame({'Mkt_RF': mkt, 'SMB': smb, 'HML': hml, 'MOM': mom}).dropna()
    return df


# %% [markdown]
# ## Regresión de factores (OLS)

# %%
def regresion_factores(rend_port, factores_df):
    """
    Regresión OLS del portafolio contra factores FF.
    Retorna DataFrame con betas, R2, alpha anual y t-stats.
    """
    data = rend_port.to_frame('Portfolio').join(factores_df).dropna()
    Y    = data['Portfolio']
    X    = sm.add_constant(data[factores_df.columns])
    res  = sm.OLS(Y, X).fit()
    out  = pd.DataFrame({
        'Beta':    res.params,
        't-stat':  res.tvalues,
        'p-value': res.pvalues
    })
    out.index.name = 'Factor'
    alpha_anual = res.params.get('const', 0) * 252
    return out, res.rsquared, alpha_anual


# %%
def fig_factor_betas(betas_pmv, betas_pms):
    """Gráfica comparativa de betas de factores PMV vs PMS."""
    factores = [i for i in betas_pmv.index if i != 'const']
    fig = go.Figure()
    for betas, lbl, col in [(betas_pmv, 'PMV', C_ERR), (betas_pms, 'PMS', C_GOLD)]:
        vals = [betas.loc[f, 'Beta'] for f in factores if f in betas.index]
        fig.add_trace(go.Bar(name=lbl, x=factores, y=vals,
                             marker_color=col, opacity=0.85,
                             text=[f'{v:.3f}' for v in vals],
                             textposition='outside'))
    fig.add_hline(y=0, line_color='#484f58')
    fig.update_yaxes(title='Beta')
    return apl(fig, 'Exposición a Factores Fama-French — PMV vs PMS', alto=400, barmode='group')
