# %% [markdown]
# # 📊 Análisis de Portafolios — Pipeline Institucional (orquestador)
#
# **Módulo 8 | Finanzas Cuantitativas con Python — Tec de Monterrey**
#
# Este script orquesta el pipeline completo importando los módulos de `src/`.
# Es la versión local (VS Code Interactive Window) del notebook institucional
# `legacy/markowitz_v3_institucional.ipynb`. Se ejecuta por bloques `# %%` y
# se puede convertir a `.ipynb` con jupytext para la entrega académica:
#
# ```bash
# jupytext --to notebook notebooks/analisis.py
# ```
#
# **Datos:** desde Fase 4 corre con **datos reales de Yahoo Finance**
# (`USAR_DATOS_SINTETICOS=False`, ventana de 6 años para que stress/drawdown
# tengan historia). Poner el flag en `True` vuelve al modo GBM offline.
# Los precios de mercado son datos públicos; lo privado (acc, P&L de IBKR)
# NO entra aquí — sigue hardcodeado en el dashboard.
#
# **Alcance (Fase 4):** pipeline analítico A–M + Fase 2 (fundamental, señales,
# noticias) + exportación a Excel/PDF y a `dashboard/data.json` (`exportar_json`).

# %%
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Hacer importable src/ (independiente del cwd donde se ejecute la celda)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in dir() else os.getcwd()
if os.path.basename(ROOT) == 'notebooks':
    ROOT = os.path.dirname(ROOT)
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)
os.chdir(ROOT)

from fetch import (descargar_precios, estadisticos_base,
                   generar_precios_sinteticos, cargar_portafolio_csv)
from optimization import (cov_ledoit_wolf, optimizar, black_litterman,
                          simular_frontera, fig_heatmap_corr, fig_frontera)
from risk import (calcular_riesgo_completo, fig_var_distribucion,
                  calcular_drawdown, fig_drawdown, rolling_metrics, fig_rolling)
from stress import PERIODOS_STRESS, stress_testing, fig_stress
from factors import descargar_factores_ff, regresion_factores, fig_factor_betas
from regime import detectar_regimenes, fig_regimenes
from projection import monte_carlo_forward, fig_monte_carlo
from rebalance import calcular_rebalanceo, fig_rebalanceo
from technical import (calcular_rsi, calcular_macd, calcular_estocastico,
                       detectar_señales, fig_tecnica, fig_base100)
from fundamental import obtener_fundamentales
from signals import generar_senales
from news import analizar_noticias
from export import exportar_excel, exportar_pdf, exportar_json

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, '.env'))

print('✅ Módulos importados desde src/')

# %% [markdown]
# ## 1. Parámetros del análisis
#
# Reemplazan a los widgets interactivos del notebook legacy. Editar aquí.

# %%
USAR_DATOS_SINTETICOS = False  # Fase 4: False = yfinance real; True = GBM offline

TICKERS, PESOS_ACTUALES = cargar_portafolio_csv(os.path.join(ROOT, 'data', 'synthetic_portfolio.csv'))
TICKERS_ORIG = list(TICKERS)   # lista original (para detectar tickers descartados → _stale)

PERIODO   = 'diaria'                     # 'diaria' | 'semanal' | 'mensual'
TASA_RF   = 0.0525                       # tasa libre de riesgo anual (decimal)
PESO_MIN  = 0.05                         # peso mínimo por activo
PESO_MAX  = 0.25                         # peso máximo por activo
BL_TAU    = 0.05                         # Black-Litterman: incertidumbre
BL_VIEWS  = 'NVDA:+0.05\nMETA:+0.03\nNVO:-0.02'
BL_CONF   = 50                           # Black-Litterman: confianza %
N_SIMS    = 10000                        # simulaciones Monte Carlo
AÑOS_MC   = [1, 3, 5]                     # horizontes Monte Carlo
NAV       = float(os.getenv('NAV_USD', 137723.85))  # capital base USD (desde .env)
COMISION  = 0.005                        # comisión por operación (decimal)

FACTOR    = {'diaria': 252, 'semanal': 52, 'mensual': 12}[PERIODO]
RF_P      = (1 + TASA_RF) ** (1 / FACTOR) - 1
RF_A      = (1 + RF_P) ** FACTOR - 1
IVL       = {'diaria': '1d', 'semanal': '1wk', 'mensual': '1mo'}[PERIODO]
FECHA_FIN = datetime.today().strftime('%Y-%m-%d')
FECHA_INI = (datetime.today() - timedelta(days=365 * 6)).strftime('%Y-%m-%d')  # 6 años

print(f'  {len(TICKERS)} activos | {PERIODO} | RF={TASA_RF:.2%} | NAV=${NAV:,.0f}')
print(f'  Modo datos: {"SINTÉTICOS (GBM)" if USAR_DATOS_SINTETICOS else "REALES (yfinance)"}')

# %% [markdown]
# ## 2. [A] Descarga de datos y estadísticos por activo

# %%
if USAR_DATOS_SINTETICOS:
    precios = generar_precios_sinteticos(TICKERS, n_dias=1260, semilla=42)
else:
    precios = descargar_precios(TICKERS, FECHA_INI, FECHA_FIN, IVL)

TICKERS = precios.columns.tolist(); N = len(TICKERS)
rend  = np.log(precios / precios.shift(1)).dropna()
stats = estadisticos_base(rend, FACTOR, RF_P)
print(stats[['Rend_Anual', 'Std_Anual', 'Sharpe', 'Skewness', 'Kurtosis']].to_string())

# Datos diarios siempre (para stress/drawdown)
if PERIODO != 'diaria' and not USAR_DATOS_SINTETICOS:
    precios_d = descargar_precios(TICKERS, FECHA_INI, FECHA_FIN, '1d')
    rend_d    = np.log(precios_d / precios_d.shift(1)).dropna()
else:
    precios_d = precios; rend_d = rend

# %% [markdown]
# ## 3. [B] Covarianza Ledoit-Wolf + correlación

# %%
corr = rend.corr()
cov_muestral = rend.cov() * FACTOR
cov_lw, shrinkage = cov_ledoit_wolf(rend, FACTOR)
print(f'  Coeficiente de shrinkage: {shrinkage:.4f}')
Sigma = cov_lw.values
mu    = stats['Rend_Anual'].values
fig_corr_plot = fig_heatmap_corr(corr)

# %% [markdown]
# ## 4. [C] Optimización: PMV · PMS · Risk Parity · Black-Litterman

# %%
pmv = optimizar(mu, Sigma, RF_A, N, PESO_MIN, PESO_MAX, 'varianza')
pms = optimizar(mu, Sigma, RF_A, N, PESO_MIN, PESO_MAX, 'sharpe')
prp = optimizar(mu, Sigma, RF_A, N, PESO_MIN, PESO_MAX, 'risk_parity')

# Black-Litterman: retornos de equilibrio (proxy pesos iguales)
w_eq  = np.full(N, 1 / N)
lam_  = (RF_A + 0.05) / float(w_eq @ Sigma @ w_eq)
mu_eq = lam_ * Sigma @ w_eq
mu_bl = black_litterman(mu_eq, Sigma, RF_A, TICKERS, BL_VIEWS, BL_TAU, BL_CONF)
pbl   = optimizar(mu_bl, Sigma, RF_A, N, PESO_MIN, PESO_MAX, 'sharpe')
pbl['mu_bl'] = mu_bl

for p, lbl in [(pmv, 'PMV'), (pms, 'PMS'), (prp, 'Risk Parity'), (pbl, 'Black-Litterman')]:
    print(f'  {lbl:20s} Rend={p["rendimiento"]:.2%}  sigma={p["riesgo"]:.2%}  Sharpe={p["sharpe"]:.3f}')

# %% [markdown]
# ## 5. [D] Frontera eficiente + CML

# %%
df_sim, _ = simular_frontera(mu, Sigma, RF_A, N, PESO_MIN, PESO_MAX, 500)
fig_fron_plot = fig_frontera(df_sim, pmv, pms, stats, TICKERS, RF_A)

# %% [markdown]
# ## 6. [E] Riesgo avanzado: VaR / CVaR / Monte Carlo

# %%
rend_pmv_d = (rend_d * pmv['pesos'][:len(rend_d.columns)]).sum(axis=1)
rend_pms_d = (rend_d * pms['pesos'][:len(rend_d.columns)]).sum(axis=1)

riesgo_pmv = calcular_riesgo_completo(rend_pmv_d, pmv['rendimiento'], pmv['riesgo'], RF_A, 'PMV')
riesgo_pms = calcular_riesgo_completo(rend_pms_d, pms['rendimiento'], pms['riesgo'], RF_A, 'PMS')
df_riesgo  = pd.DataFrame([riesgo_pmv, riesgo_pms]).set_index('Portafolio')
print(df_riesgo.to_string())

# %% [markdown]
# ## 7. [F] Stress testing histórico

# %%
tks_d   = [t for t in TICKERS if t in precios_d.columns]
w_pmv_d = np.array([pmv['pesos'][TICKERS.index(t)] for t in tks_d])
w_pms_d = np.array([pms['pesos'][TICKERS.index(t)] for t in tks_d])
w_pmv_d /= w_pmv_d.sum(); w_pms_d /= w_pms_d.sum()

df_stress = stress_testing(precios_d[tks_d], w_pmv_d, w_pms_d, tks_d)
print(df_stress.to_string())

# %% [markdown]
# ## 8. [G] Drawdown analysis (underwater)

# %%
dd_pmv, max_dd_pmv, calmar_pmv, rec_pmv = calcular_drawdown(rend_pmv_d)
dd_pms, max_dd_pms, calmar_pms, rec_pms = calcular_drawdown(rend_pms_d)
print(f'  PMV — Max DD: {max_dd_pmv:.2%}  Calmar: {calmar_pmv:.2f}  Rec: {rec_pmv} dias')
print(f'  PMS — Max DD: {max_dd_pms:.2%}  Calmar: {calmar_pms:.2f}  Rec: {rec_pms} dias')
fig_dd_plot = fig_drawdown(rend_pmv_d, rend_pms_d, rend_d, TICKERS)

# %% [markdown]
# ## 9. [H] Rolling analysis (Sharpe / volatilidad)

# %%
roll_pmv = rolling_metrics(rend_pmv_d, 252, RF_A)
roll_pms = rolling_metrics(rend_pms_d, 252, RF_A)
fig_roll_sh = fig_rolling(roll_pmv, roll_pms, 'sharpe')
fig_roll_vo = fig_rolling(roll_pmv, roll_pms, 'vol')

# %% [markdown]
# ## 10. [I] Factores Fama-French
#
# Con datos sintéticos se generan factores GBM offline; con datos reales se
# descargan los ETFs proxy (SPY/IWM/IVE/IWF/MTUM/USMV/BIL).

# %%
if USAR_DATOS_SINTETICOS:
    rng = np.random.default_rng(7)
    factores = pd.DataFrame(
        rng.normal(0, 0.01, (len(rend_d), 4)),
        index=rend_d.index, columns=['Mkt_RF', 'SMB', 'HML', 'MOM'])
else:
    factores = descargar_factores_ff(FECHA_INI, FECHA_FIN)

alpha_pmv = alpha_pms = None; betas_pmv_df = betas_pms_df = None
r2_pmv = r2_pms = None
if factores is not None:
    betas_pmv_df, r2_pmv, alpha_pmv = regresion_factores(rend_pmv_d, factores)
    betas_pms_df, r2_pms, alpha_pms = regresion_factores(rend_pms_d, factores)
    print(f'  PMV — R2={r2_pmv:.3f}  Alpha anual={alpha_pmv:.2%}')
    print(f'  PMS — R2={r2_pms:.3f}  Alpha anual={alpha_pms:.2%}')
    fig_ff_plot = fig_factor_betas(betas_pmv_df, betas_pms_df)
else:
    fig_ff_plot = None
    print('  No se pudieron descargar factores FF')

# %% [markdown]
# ## 11. [J] Régimen de mercado (K-Means)

# %%
rend_pms_port = pd.Series((rend_d.reindex(columns=tks_d) * w_pms_d).sum(axis=1), name='PMS')
reg_df, reg_stats = detectar_regimenes(rend_pms_port)
print(reg_stats.to_string())
cum_port = np.exp(rend_pms_port.cumsum()) * 100
reg_df_aligned = reg_df.reindex(rend_pms_port.index).ffill()
fig_reg_plot = fig_regimenes(cum_port, reg_df_aligned)

# %% [markdown]
# ## 12. [K] Monte Carlo forward-looking (GBM)

# %%
mc_pmv = monte_carlo_forward(pmv['rendimiento'], pmv['riesgo'], NAV, N_SIMS, AÑOS_MC, 'PMV')
mc_pms = monte_carlo_forward(pms['rendimiento'], pms['riesgo'], NAV, N_SIMS, AÑOS_MC, 'PMS')
for años in AÑOS_MC:
    for mc, lbl in [(mc_pmv, 'PMV'), (mc_pms, 'PMS')]:
        r = mc[años]
        print(f'  {lbl} {años}Y — Med: ${r["p50"]:,.0f}  P5: ${r["p5"]:,.0f}  '
              f'P95: ${r["p95"]:,.0f}  Prob>0: {r["prob_positivo"]:.1%}  Prob 2x: {r["prob_2x"]:.1%}')
fig_mc_plot = fig_monte_carlo(mc_pms, NAV, AÑOS_MC, 'PMS')

# MC summary rows (para Excel/PDF)
mc_rows = []
for años in AÑOS_MC:
    for mc, lbl in [(mc_pmv, 'PMV'), (mc_pms, 'PMS')]:
        r = mc[años]
        mc_rows.append({'Port': lbl, 'Anos': años, 'P5': r['p5'], 'P25': r['p25'],
                        'P50': r['p50'], 'P75': r['p75'], 'P95': r['p95'],
                        'Prob_positivo': r['prob_positivo'], 'Prob_2x': r['prob_2x']})

# %% [markdown]
# ## 13. [L] Trade list de rebalanceo con costos

# %%
precios_actuales = precios_d.iloc[-1].to_dict()
df_reb_pms = calcular_rebalanceo(TICKERS, pms['pesos'], precios_actuales, NAV, COMISION)
df_reb_pmv = calcular_rebalanceo(TICKERS, pmv['pesos'], precios_actuales, NAV, COMISION)
print(df_reb_pms[['Peso Actual', 'Peso Objetivo', 'Gap', 'Dirección',
                  'USD Trade', 'Comisión Est.']].to_string())
print(f'  Turnover total: ${df_reb_pms["USD Trade"].sum():,.0f} '
      f'({df_reb_pms["USD Trade"].sum()/NAV:.1%} del NAV)')
fig_reb_plot = fig_rebalanceo(df_reb_pms, 'Rebalanceo hacia PMS (Max Sharpe)')

# %% [markdown]
# ## 14. [M] Análisis técnico (SMA / RSI / MACD / Estocástico)

# %%
inds = {}
for tk in TICKERS:
    if tk not in precios_d.columns: continue
    c = precios_d[tk].dropna()
    v = pd.Series(dtype=float)
    if not USAR_DATOS_SINTETICOS:
        try:
            import yfinance as yf
            raw = yf.download(tk, start=FECHA_INI, end=FECHA_FIN, interval='1d',
                              progress=False, auto_adjust=False)
            if 'Volume' in raw.columns: v = raw['Volume'].squeeze()
        except Exception:
            pass
    rsi_s = calcular_rsi(c); ml, ms, mh = calcular_macd(c); pk, pd_ = calcular_estocastico(c)
    inds[tk] = {'close': c, 'volume': v, 'sma50': c.rolling(50).mean(),
                'sma100': c.rolling(100).mean(), 'sma200': c.rolling(200).mean(),
                'rsi': rsi_s, 'macd': ml, 'macd_s': ms, 'macd_h': mh, 'pct_k': pk, 'pct_d': pd_,
                'señales': detectar_señales(c, rsi_s, ml, ms)}

fig_tec_plot = fig_tecnica(TICKERS[0], inds[TICKERS[0]])
fig_b100_plot = fig_base100(inds, TICKERS)
print(f'  Indicadores calculados para {len(inds)} activos')

# %% [markdown]
# ## 15. [Fase 4] Posiciones: precio real + retornos (YTD / 1m / 1y)
#
# Precio de cierre real y retornos por ticker desde `precios` (yfinance).
# Si yfinance no devolvió datos para un ticker se marca `_stale: true` sin
# romper el pipeline (manejo en el orquestador, sin tocar `fetch.py`).
# Los campos privados (acciones y P&L de IBKR) NO se calculan aquí.

# %%
def _ret_back(serie, n):
    """Retorno % respecto a `n` barras hacia atrás (None si no hay historia)."""
    s = serie.dropna()
    if len(s) > n:
        return float(s.iloc[-1] / s.iloc[-1 - n] - 1) * 100
    return None

posiciones = {}
for tk in TICKERS_ORIG:
    if tk in precios.columns:
        s = precios[tk].dropna()
        r1m, r1y = _ret_back(s, 21), _ret_back(s, 252)
        anio = s.index[-1].year
        s_ytd = s[s.index.year == anio]
        ytd = float(s.iloc[-1] / s_ytd.iloc[0] - 1) * 100 if len(s_ytd) > 1 else None
        posiciones[tk] = {
            'precio_actual': round(float(s.iloc[-1]), 4),
            'retorno_ytd':   round(ytd, 2) if ytd is not None else None,
            'retorno_1m':    round(r1m, 2) if r1m is not None else None,
            'retorno_1y':    round(r1y, 2) if r1y is not None else None,
            '_stale':        False,
        }
    else:
        posiciones[tk] = {'precio_actual': None, 'retorno_ytd': None,
                          'retorno_1m': None, 'retorno_1y': None, '_stale': True}

_stale_tks = [t for t, p in posiciones.items() if p['_stale']]
print(f'  Posiciones: {len(posiciones)} tickers | sin datos (_stale): {_stale_tks or "ninguno"}')

# %% [markdown]
# ## 16. [Fase 2] Fundamental · Señales swing · Noticias (Gemini)

# %%
print('  [j] Fundamentales (yfinance .info)...')
fund = obtener_fundamentales(TICKERS)
print('  [k] Señales swing (triple-barrier)...')
sen = generar_senales(precios)
print(f'      {len(sen)} señales generadas')
print('  [l] Noticias + análisis Gemini...')
noti = analizar_noticias(TICKERS)

# %% [markdown]
# ## 17. [O/P] Exportación a Excel y PDF
#
# Se arma el dict de resultados `R` y se generan los archivos en `outputs/`.
# (El dashboard HTML — Módulo N del legacy — queda fuera de alcance esta fase.)

# %%
R = {
    'TICKERS': TICKERS, 'N': N, 'AÑOS_MC': AÑOS_MC, 'N_SIMS': N_SIMS,
    'NAV': NAV, 'RF_A': RF_A, 'shrinkage': shrinkage,
    'FECHA_INI': FECHA_INI, 'FECHA_FIN': FECHA_FIN,
    'PESO_MIN': PESO_MIN, 'PESO_MAX': PESO_MAX, 'BL_TAU': BL_TAU, 'BL_CONF': BL_CONF,
    'precios': precios, 'rend': rend, 'stats': stats, 'corr': corr, 'cov_lw': cov_lw,
    'df_sim': df_sim, 'df_riesgo': df_riesgo, 'df_stress': df_stress,
    'df_reb_pms': df_reb_pms, 'df_reb_pmv': df_reb_pmv, 'reg_stats': reg_stats,
    'pmv': pmv, 'pms': pms, 'prp': prp, 'pbl': pbl,
    'mc_rows': mc_rows, 'mc_pmv': mc_pmv, 'mc_pms': mc_pms,
    'riesgo_pms': riesgo_pms,
    'max_dd_pmv': max_dd_pmv, 'max_dd_pms': max_dd_pms,
    'calmar_pmv': calmar_pmv, 'calmar_pms': calmar_pms,
    'rec_pmv': rec_pmv, 'rec_pms': rec_pms,
    'betas_pmv_df': betas_pmv_df, 'betas_pms_df': betas_pms_df,
    'inds': inds,
    'posiciones': posiciones,
    'figs': {
        'frontera': fig_fron_plot, 'corr': fig_corr_plot, 'stress': fig_stress(df_stress),
        'dd': fig_dd_plot, 'rolling': fig_roll_sh, 'mc': fig_mc_plot,
    },
}
if fig_ff_plot is not None:
    R['figs']['ff'] = fig_ff_plot

ruta_excel, hojas = exportar_excel(R)
ruta_pdf = exportar_pdf(R)

# %% [markdown]
# ## 18. [Fase 4] Exportación a dashboard/data.json
#
# Escribe el JSON que consume `dashboard/index.html`: base analítica + posiciones
# (precio real) + fundamental + señales + noticias.

# %%
ruta_json = exportar_json(R, fundamental=fund, senales=sen, noticias=noti,
                          ruta=os.path.join(ROOT, 'dashboard', 'data.json'))

print('\n' + '=' * 62)
print('  ANÁLISIS INSTITUCIONAL COMPLETO')
print(f'  Excel : {ruta_excel}  ({hojas})')
print(f'  PDF   : {ruta_pdf}')
print(f'  JSON  : {ruta_json}')
print('=' * 62)
