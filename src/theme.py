# %% [markdown]
# # theme — Tema visual global (Plotly dark financiero)
#
# Módulo compartido por todos los módulos de gráficas. Contiene la
# configuración de presentación extraída tal cual del notebook legacy
# (`pio.templates.default='plotly_dark'`, `PLOTLY_LAYOUT`, paleta de
# colores y el helper `apl()`).
#
# **No contiene lógica financiera** — es únicamente estilo. Se separó en
# su propio archivo para no duplicar estas constantes en cada módulo.

# %%
import plotly.express as px
import plotly.io as pio

# ─── Tema dark financiero global ─────────────────────────────────────────────
pio.templates.default = 'plotly_dark'
PLOTLY_LAYOUT = dict(
    paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
    font=dict(family='Inter,Segoe UI,system-ui', color='#c9d1d9', size=12),
    title_font=dict(size=16, color='#58a6ff'),
    legend=dict(bgcolor='rgba(22,27,34,0.9)', bordercolor='#30363d',
                borderwidth=1, font=dict(size=11)),
    xaxis=dict(gridcolor='#21262d', zeroline=False, linecolor='#30363d'),
    yaxis=dict(gridcolor='#21262d', zeroline=False, linecolor='#30363d'),
    margin=dict(l=65, r=45, t=75, b=55)
)
COLORES = px.colors.qualitative.Bold
C_OK    = '#3fb950'
C_ERR   = '#ff7b72'
C_GOLD  = '#ffd700'
C_BLUE  = '#58a6ff'
C_WARN  = '#ffa657'


# %%
def apl(fig, titulo='', alto=500, **kw):
    """Aplica el layout dark global a una figura Plotly y la retorna."""
    lay = dict(**PLOTLY_LAYOUT); lay['height'] = alto
    if titulo:
        lay['title'] = dict(text=titulo, font=dict(size=16, color=C_BLUE), x=0.02)
    lay.update(kw); fig.update_layout(**lay); return fig
