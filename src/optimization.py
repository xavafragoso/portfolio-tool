# %% [markdown]
# # optimization — Covarianza, optimizadores y frontera eficiente
#
# Módulos B, C y D del notebook legacy:
# - Ledoit-Wolf shrinkage de la covarianza
# - Optimizador SLSQP (Min Varianza / Max Sharpe / Risk Parity)
# - Black-Litterman
# - Frontera eficiente (Monte Carlo de pesos) + CML
# - Heatmap de correlación
#
# Lógica y fórmulas idénticas al legacy.

# %%
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from tqdm.auto import tqdm

import plotly.graph_objects as go

from theme import apl, COLORES, C_OK, C_ERR, C_GOLD, C_BLUE, C_WARN


# %% [markdown]
# ## Covarianza con Ledoit-Wolf shrinkage

# %%
def cov_ledoit_wolf(rend, factor):
    """
    Estimación robusta de covarianza via Ledoit-Wolf shrinkage.
    Reduce el error de estimación de la matriz muestral.
    Retorna cov anualizada y coeficiente de shrinkage aplicado.
    """
    lw = LedoitWolf().fit(rend.values)
    cov_lw = pd.DataFrame(lw.covariance_ * factor,
                          index=rend.columns, columns=rend.columns)
    return cov_lw, lw.shrinkage_


# %% [markdown]
# ## Optimizadores (SLSQP): Min Varianza · Max Sharpe · Risk Parity

# %%
def optimizar(mu, Sigma, rf, N, w_min, w_max, obj='sharpe'):
    """Optimizador SLSQP genérico: obj = 'sharpe' | 'varianza' | 'risk_parity'."""
    bounds = [(w_min, w_max)] * N
    cons   = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    w0     = np.full(N, 1 / N)
    if obj == 'varianza':
        fun = lambda w: float(w @ Sigma @ w)
    elif obj == 'risk_parity':
        def fun(w):
            var  = float(w @ Sigma @ w)
            mrc  = Sigma @ w            # Marginal Risk Contribution
            rc   = w * mrc / np.sqrt(var)  # Risk Contribution
            rc_t = np.sqrt(var) / N     # Target: igual contribución
            return np.sum((rc - rc_t) ** 2)
    else:  # sharpe
        fun = lambda w: -(float(w @ mu) - rf) / np.sqrt(float(w @ Sigma @ w))
    res = minimize(fun, w0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 3000, 'ftol': 1e-13})
    w = res.x; r = float(w @ mu); s = np.sqrt(float(w @ Sigma @ w))
    return {'pesos': w, 'rendimiento': r, 'riesgo': s, 'sharpe': (r - rf) / s,
            'conv': res.success}


# %% [markdown]
# ## Black-Litterman

# %%
def black_litterman(mu_eq, Sigma, rf, tickers, views_str, tau, conf_pct):
    """
    Black-Litterman: combina retornos de equilibrio con views del PM.

    Parámetros
    ----------
    mu_eq    : retornos implícitos de equilibrio (CAPM proxy: Sigma@w_mkt * lambda_)
    Sigma    : matriz de covarianza anualizada
    views_str: string 'TICKER:view\nTICKER2:view2'
    tau      : escalar de incertidumbre (típico 0.01-0.10)
    conf_pct : confianza en los views (10-100%)
    """
    n = len(tickers)
    # Parsear views
    P_rows, Q_vals = [], []
    for line in views_str.strip().split('\n'):
        line = line.strip()
        if ':' not in line: continue
        tk, val = line.split(':', 1)
        tk = tk.strip().upper()
        if tk not in tickers: continue
        p = np.zeros(n)
        p[tickers.index(tk)] = 1.0
        P_rows.append(p)
        Q_vals.append(float(val))
    if not P_rows:
        print('  BL: sin views válidos — usando equilibrio puro')
        return mu_eq
    P = np.array(P_rows)
    Q = np.array(Q_vals)
    # Omega: matriz de incertidumbre de views (proporcional a Sigma)
    omega_scale = 1 / (conf_pct / 100 + 1e-10)
    Omega = np.diag(np.diag(P @ (tau * Sigma) @ P.T)) * omega_scale
    # Fórmula BL
    tSigma = tau * Sigma
    M1 = np.linalg.inv(tSigma)
    M2 = P.T @ np.linalg.inv(Omega) @ P
    mu_bl = np.linalg.inv(M1 + M2) @ (M1 @ mu_eq + P.T @ np.linalg.inv(Omega) @ Q)
    return mu_bl


# %% [markdown]
# ## Frontera eficiente (simulación de pesos)

# %%
def simular_frontera(mu, Sigma, rf, N, w_min, w_max, n_sim=500):
    """Genera `n_sim` portafolios aleatorios (Dirichlet recortado) para trazar
    la nube de la frontera eficiente. Retorna (DataFrame métricas, lista pesos)."""
    sr, ss, sh, sw = [], [], [], []
    pb = tqdm(total=n_sim, desc='Frontera')
    while len(sr) < n_sim:
        w = np.clip(np.random.dirichlet(np.ones(N)), w_min, w_max)
        w /= w.sum()
        r = float(w @ mu); s = np.sqrt(float(w @ Sigma @ w))
        sr.append(r); ss.append(s); sh.append((r - rf) / s); sw.append(w); pb.update(1)
    pb.close()
    df = pd.DataFrame({'Rendimiento': sr, 'Riesgo': ss, 'Sharpe': sh})
    return df, sw


# %% [markdown]
# ## Gráficas: heatmap de correlación y frontera + CML

# %%
def fig_heatmap_corr(corr):
    """Heatmap de la matriz de correlación de rendimientos logarítmicos."""
    n = len(corr); txt = [[f'{v:.2f}' for v in row] for row in corr.values]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        text=txt, texttemplate='%{text}', textfont=dict(size=max(7, 11 - n // 3)),
        colorscale=[[0, '#d73027'], [0.25, '#f46d43'], [0.5, '#1a1a2e'],
                    [0.75, '#4575b4'], [1, '#313695']],
        zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title='rho', thickness=14, len=0.8, tickfont=dict(color='#c9d1d9'))))
    return apl(fig, 'Matriz de Correlacion — Rendimientos Logaritmicos', alto=max(420, 48 * n))


# %%
def fig_frontera(df_sim, pmv, pms, stats, tickers, rf_a):
    """Frontera eficiente de Markowitz con CML, PMV, PMS y activos individuales."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_sim['Riesgo'], y=df_sim['Rendimiento'], mode='markers',
        marker=dict(color=df_sim['Sharpe'], colorscale='Viridis', size=5, opacity=0.6,
                    colorbar=dict(title='Sharpe', thickness=12, tickfont=dict(color='#c9d1d9'))),
        hovertemplate='sigma:%{x:.2%}<br>mu:%{y:.2%}<br>Sharpe:%{marker.color:.3f}',
        name='Simulados'))
    sig_r = np.linspace(0, df_sim['Riesgo'].max() * 1.1, 200)
    fig.add_trace(go.Scatter(x=sig_r, y=rf_a + pms['sharpe'] * sig_r,
                             mode='lines', line=dict(color=C_BLUE, width=2, dash='dot'), name='CML'))
    for pt, lbl, col in [(pmv, 'PMV', C_ERR), (pms, 'PMS', C_GOLD)]:
        fig.add_trace(go.Scatter(
            x=[pt['riesgo']], y=[pt['rendimiento']], mode='markers+text',
            marker=dict(color=col, size=18, symbol='star', line=dict(width=2, color='white')),
            text=[lbl], textposition='top right', textfont=dict(color=col, size=13),
            name=f'{lbl} Sharpe={pt["sharpe"]:.2f}'))
    for i, tk in enumerate(tickers):
        fig.add_trace(go.Scatter(
            x=[stats.loc[tk, 'Std_Anual']], y=[stats.loc[tk, 'Rend_Anual']],
            mode='markers+text',
            marker=dict(color=COLORES[i % len(COLORES)], size=10, symbol='diamond',
                        line=dict(width=1, color='white')),
            text=[tk], textposition='top center', textfont=dict(size=10), showlegend=False))
    fig.update_xaxes(title='Riesgo sigma anualizado', tickformat='.0%')
    fig.update_yaxes(title='Rendimiento anual', tickformat='.0%')
    return apl(fig, 'Frontera Eficiente de Markowitz + CML', alto=600)
