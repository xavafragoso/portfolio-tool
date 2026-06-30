# %% [markdown]
# # projection — Monte Carlo forward-looking
#
# Módulo K del notebook legacy. Simulación GBM (Geometric Brownian Motion)
# forward-looking para horizontes de 1/3/5 años, con percentiles y
# probabilidades de outcome.
#
# Lógica y fórmulas idénticas al legacy.

# %%
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from theme import apl, C_OK, C_ERR, C_BLUE, C_WARN


# %% [markdown]
# ## Simulación Monte Carlo (GBM)

# %%
def monte_carlo_forward(mu_a, sig_a, nav, n_sims, años_list, label):
    """
    Simulación GBM (Geometric Brownian Motion) forward-looking.
    Genera distribución de outcomes para cada horizonte temporal.
    Retorna percentiles clave: 5, 25, 50, 75, 95.
    """
    mu_d  = mu_a / 252
    sig_d = sig_a / np.sqrt(252)
    resultados = {}

    for años in años_list:
        dias = int(años * 252)
        # GBM: S_t = S_0 * exp((mu - 0.5*sig^2)*t + sig*sqrt(t)*Z)
        shocks = np.random.normal(0, 1, (n_sims, dias))
        log_ret = (mu_d - 0.5 * sig_d ** 2) + sig_d * shocks
        cum_ret = np.exp(log_ret.cumsum(axis=1))
        final   = cum_ret[:, -1] * nav

        resultados[años] = {
            'paths_sample': cum_ret[:100] * nav,  # 100 trayectorias para graficar
            'final': final,
            'p5':   float(np.percentile(final, 5)),
            'p25':  float(np.percentile(final, 25)),
            'p50':  float(np.percentile(final, 50)),
            'p75':  float(np.percentile(final, 75)),
            'p95':  float(np.percentile(final, 95)),
            'prob_positivo': float(np.mean(final > nav)),
            'prob_2x':       float(np.mean(final > 2 * nav)),
        }
    return resultados


# %%
def fig_monte_carlo(mc_results, nav, años_list, label):
    """Gráficas de Monte Carlo: trayectorias + distribución final por horizonte."""
    fig = make_subplots(
        rows=len(años_list), cols=2,
        subplot_titles=[
            item for a in años_list
            for item in [f'Trayectorias {a}Y — {label}', f'Distribución Final {a}Y']
        ],
        horizontal_spacing=0.08, vertical_spacing=0.12
    )
    for i, años in enumerate(años_list):
        res = mc_results[años]
        dias = int(años * 252)
        eje_x = list(range(dias + 1))

        # Trayectorias (sample)
        for j, path in enumerate(res['paths_sample'][:50]):
            fig.add_trace(go.Scatter(
                x=eje_x[1:], y=path,
                line=dict(color=C_BLUE, width=0.4), opacity=0.3,
                showlegend=False
            ), row=i + 1, col=1)

        # Percentiles
        for pct, col_p, nm in [('p95', C_OK, 'p95'), ('p75', C_WARN, 'p75'),
                                ('p50', C_BLUE, 'Mediana'),
                                ('p25', C_ERR, 'p25'), ('p5', '#ff0000', 'p5')]:
            val_f = res[pct]
            # Solo dibujar como línea horizontal en el punto final
            fig.add_trace(go.Scatter(
                x=[1, dias], y=[nav, val_f],
                line=dict(color=col_p, width=1.8, dash='dash'),
                name=f'{nm}: ${val_f:,.0f}', showlegend=(i == 0)
            ), row=i + 1, col=1)
        fig.add_hline(y=nav, line_dash='dot', line_color='#484f58', row=i + 1, col=1)

        # Histograma distribución final
        fig.add_trace(go.Histogram(
            x=res['final'], nbinsx=60,
            marker_color=C_BLUE, opacity=0.7, showlegend=False,
            name=f'Dist {años}Y'
        ), row=i + 1, col=2)
        fig.add_vline(x=nav, line_dash='dash', line_color=C_WARN, row=i + 1, col=2)
        fig.add_vline(x=res['p50'], line_dash='dash', line_color=C_OK, row=i + 1, col=2)

        fig.update_xaxes(gridcolor='#21262d', row=i + 1, col=1)
        fig.update_xaxes(gridcolor='#21262d', title='USD Final', row=i + 1, col=2)
        fig.update_yaxes(gridcolor='#21262d', title='USD', row=i + 1, col=1)
        fig.update_yaxes(gridcolor='#21262d', row=i + 1, col=2)

    return apl(fig, f'Monte Carlo GBM — Proyección Forward-Looking | {label}',
               alto=380 * len(años_list))
