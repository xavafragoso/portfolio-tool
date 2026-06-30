# %% [markdown]
# # risk — Riesgo avanzado, drawdown y rolling analysis
#
# Módulos E, G y H del notebook legacy:
# - VaR paramétrico / histórico / Monte Carlo, CVaR (Expected Shortfall)
# - Drawdown, Calmar, tiempo de recuperación (underwater)
# - Rolling Sharpe / volatilidad / retorno (1Y/2Y/3Y)
#
# Lógica y fórmulas idénticas al legacy.

# %%
import numpy as np
import pandas as pd
from scipy.stats import norm

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from theme import apl, COLORES, C_OK, C_ERR, C_GOLD, C_BLUE, C_WARN


# %% [markdown]
# ## E — Riesgo avanzado: VaR, CVaR, Monte Carlo

# %%
def calcular_riesgo_completo(rend_port_diario, mu_a, sig_a, rf_a, label):
    """
    Calcula el stack completo de métricas de riesgo para un portafolio.

    Incluye: VaR paramétrico, VaR histórico, CVaR (Expected Shortfall),
    VaR Monte Carlo, todos al 95% y 99%.
    """
    Z95, Z99 = norm.ppf(0.95), norm.ppf(0.99)
    mu_d  = mu_a / 252; sig_d = sig_a / np.sqrt(252)

    # VaR paramétrico
    var95_p = -(mu_d - Z95 * sig_d)
    var99_p = -(mu_d - Z99 * sig_d)

    # VaR histórico (percentil de pérdidas reales)
    r = rend_port_diario.dropna()
    var95_h = float(-np.percentile(r, 5))
    var99_h = float(-np.percentile(r, 1))

    # CVaR / Expected Shortfall (pérdida esperada DADO que supera el VaR)
    cvar95 = float(-r[r <= -var95_h].mean()) if (-r <= var95_h * -1).any() else var95_h
    cvar99 = float(-r[r <= -var99_h].mean()) if (-r <= var99_h * -1).any() else var99_h
    # Forma correcta
    thresh95 = np.percentile(r, 5)
    thresh99 = np.percentile(r, 1)
    cvar95   = float(-r[r <= thresh95].mean())
    cvar99   = float(-r[r <= thresh99].mean())

    # VaR Monte Carlo (10,000 simulaciones diarias)
    mc_r = np.random.normal(mu_d, sig_d, 100000)
    var95_mc = float(-np.percentile(mc_r, 5))
    var99_mc = float(-np.percentile(mc_r, 1))

    return {
        'Portafolio':     label,
        'VaR 95% Param':  var95_p,
        'VaR 99% Param':  var99_p,
        'VaR 95% Hist':   var95_h,
        'VaR 99% Hist':   var99_h,
        'CVaR 95% (ES)':  cvar95,
        'CVaR 99% (ES)':  cvar99,
        'VaR 95% MC':     var95_mc,
        'VaR 99% MC':     var99_mc,
        'VaR 99% Mensual': -(mu_d * 21 - Z99 * sig_d * np.sqrt(21)),
        'VaR 99% Anual':  -(mu_a - Z99 * sig_a),
    }


# %%
def fig_var_distribucion(rend_port, label):
    """Histograma de retornos con VaR y CVaR anotados."""
    r = rend_port.dropna()
    var95 = np.percentile(r, 5)
    var99 = np.percentile(r, 1)
    cvar99 = r[r <= var99].mean()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=r, nbinsx=80, name='Retornos',
        marker_color=C_BLUE, opacity=0.7,
        histnorm='probability density'
    ))
    # Curva normal teórica
    x_n = np.linspace(r.min(), r.max(), 300)
    y_n = norm.pdf(x_n, r.mean(), r.std())
    fig.add_trace(go.Scatter(x=x_n, y=y_n, name='Normal teórica',
                             line=dict(color=C_WARN, width=2)))
    # Líneas VaR
    for xv, col, nm in [(var95, C_ERR, 'VaR 95%'), (var99, '#ff0000', 'VaR 99%'), (cvar99, C_GOLD, 'CVaR 99%')]:
        fig.add_vline(x=xv, line_dash='dash', line_color=col,
                      annotation_text=nm, annotation_position='top right',
                      annotation_font_color=col)
    fig.update_xaxes(title='Retorno diario', tickformat='.1%')
    fig.update_yaxes(title='Densidad')
    return apl(fig, f'Distribución de Retornos — {label} | Fat Tails vs Normal', alto=420)


# %% [markdown]
# ## G — Drawdown analysis (Max DD, Calmar, recuperación)

# %%
def calcular_drawdown(rend_serie):
    """
    Retorna serie de drawdown, max drawdown, Calmar ratio y tiempo de recuperación.
    """
    cumret  = np.exp(rend_serie.cumsum())
    peak    = cumret.cummax()
    dd      = (cumret - peak) / peak        # Drawdown serie
    max_dd  = float(dd.min())

    # Tiempo de recuperación (días desde máx drawdown hasta siguiente nuevo máx)
    idx_trough = dd.idxmin()
    after_trough = cumret.loc[idx_trough:]
    peak_before  = float(peak.loc[idx_trough])
    recovered    = after_trough[after_trough >= peak_before]
    recovery_days = (recovered.index[0] - idx_trough).days if len(recovered) > 0 else None

    # Calmar ratio: retorno anual / abs(max drawdown)
    ret_anual = float(rend_serie.mean() * 252)
    calmar    = ret_anual / abs(max_dd) if max_dd != 0 else np.nan

    # Underwater plot data
    return dd, max_dd, calmar, recovery_days


# %%
def fig_drawdown(rend_pmv, rend_pms, tickers_rend=None, tickers=None):
    """Underwater chart con PMV, PMS y activos individuales (opcional)."""
    fig = go.Figure()
    for rp, lbl, col in [(rend_pmv, 'PMV', C_ERR), (rend_pms, 'PMS', C_GOLD)]:
        dd, _, _, _ = calcular_drawdown(rp)
        # FIX (port): el legacy hacía col.replace('#','rgba(').replace(')',',0.15)')
        # que generaba un color malformado ('rgba(ff7b72'); Plotly >=6 lo rechaza.
        # Se conserva la intención original (relleno translúcido del color de línea).
        fc = f'rgba({int(col[1:3],16)},{int(col[3:5],16)},{int(col[5:7],16)},0.15)' if col.startswith('#') else col
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd, name=lbl, fill='tozeroy',
            line=dict(color=col, width=1.5),
            fillcolor=fc,
            hovertemplate=f'{lbl}: %{{y:.2%}}'
        ))
    if tickers_rend is not None and tickers is not None:
        for i, tk in enumerate(tickers[:5]):  # Top 5 para no saturar
            dd, _, _, _ = calcular_drawdown(tickers_rend[tk])
            fig.add_trace(go.Scatter(
                x=dd.index, y=dd, name=tk, line=dict(color=COLORES[i], width=1, dash='dot'),
                opacity=0.5, hovertemplate=f'{tk}: %{{y:.2%}}'
            ))
    fig.update_yaxes(title='Drawdown', tickformat='.0%')
    fig.update_xaxes(title='Fecha')
    return apl(fig, 'Underwater Chart — Drawdown Histórico', alto=430)


# %% [markdown]
# ## H — Rolling analysis (Sharpe / volatilidad / retorno)

# %%
def rolling_metrics(rend_port, factor, rf_a, ventanas_dias=[252, 504, 756]):
    """
    Calcula Sharpe, volatilidad y retorno rolling en múltiples ventanas.
    Ventanas: 252d (1Y), 504d (2Y), 756d (3Y).
    """
    resultados = {}
    rf_d = (1 + rf_a) ** (1 / factor) - 1
    for v in ventanas_dias:
        mu_r  = rend_port.rolling(v).mean() * factor
        sig_r = rend_port.rolling(v).std(ddof=1) * np.sqrt(factor)
        sh_r  = (mu_r - rf_a) / sig_r
        resultados[f'{v // 252}Y'] = {'sharpe': sh_r, 'vol': sig_r, 'rend': mu_r}
    return resultados


# %%
def fig_rolling(rolling_pmv, rolling_pms, metrica='sharpe'):
    """Gráfica de métricas rolling con subplots por ventana."""
    ventanas = list(rolling_pmv.keys())
    fig = make_subplots(rows=len(ventanas), cols=1,
                        shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=[f'Ventana {v}' for v in ventanas])
    titulos = {'sharpe': 'Sharpe Ratio Rolling', 'vol': 'Volatilidad Rolling (anual)',
               'rend': 'Retorno Rolling (anual)'}
    for i, v in enumerate(ventanas):
        for data, lbl, col in [(rolling_pmv[v], 'PMV', C_ERR), (rolling_pms[v], 'PMS', C_GOLD)]:
            serie = data[metrica]
            fig.add_trace(go.Scatter(
                x=serie.index, y=serie,
                name=f'{lbl} {v}', line=dict(color=col, width=1.5),
                hovertemplate=f'{lbl}: %{{y:.3f}}'
            ), row=i + 1, col=1)
        if metrica in ('vol', 'rend'):
            fig.update_yaxes(tickformat='.0%', row=i + 1, col=1)
        fig.update_xaxes(gridcolor='#21262d', row=i + 1, col=1)
        fig.update_yaxes(gridcolor='#21262d', row=i + 1, col=1)
    return apl(fig, titulos.get(metrica, 'Rolling'), alto=550)
