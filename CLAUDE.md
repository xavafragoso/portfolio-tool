# CLAUDE.md — Portfolio Tool

Guía para Claude Code (y para humanos) sobre este repositorio.

## Qué es

Sistema de análisis cuantitativo de portafolios migrado desde un notebook
institucional de Google Colab (`legacy/markowitz_v3_institucional.ipynb`, Módulo
6) a un proyecto local estructurado. Proyecto académico — **Módulo 8, Tec de
Monterrey**. **Solo se usan datos sintéticos** por privacidad.

> Nota: el prompt original se refería al notebook como `legacy/portfolio_v3.ipynb`;
> el archivo real en `legacy/` es `markowitz_v3_institucional.ipynb` (único notebook
> presente, se tomó ese).

## Arquitectura

```
.
├── data/synthetic_portfolio.csv   # 15 tickers + pesos aleatorios (suma 100%) — solo pruebas
├── src/                           # módulos analíticos (un archivo por dominio)
│   ├── theme.py                   # [añadido] tema Plotly global + apl() + colores (presentación, sin lógica)
│   ├── fetch.py                   # descarga + estadísticos + generador sintético
│   ├── optimization.py            # correlación, Ledoit-Wolf, Markowitz, BL, Risk Parity, frontera+CML
│   ├── risk.py                    # VaR/CVaR, drawdown/Calmar, rolling
│   ├── stress.py                  # stress testing histórico
│   ├── factors.py                 # regresión Fama-French
│   ├── regime.py                  # régimen de mercado (K-Means)
│   ├── projection.py              # Monte Carlo forward-looking (GBM)
│   ├── rebalance.py               # trade list de rebalanceo con costos
│   ├── technical.py               # SMA/RSI/MACD/Estocástico
│   └── export.py                  # Excel + PDF institucional (_t, PortfolioPDF)
├── notebooks/analisis.py          # orquestador (# %%), convertible a .ipynb con jupytext
├── dashboard/                     # vacío — dashboard nuevo es fase siguiente
└── outputs/                       # artefactos generados (gitignored)
```

- Todos los archivos `.py` usan marcadores de celda `# %%` para el Interactive
  Window de VS Code. Funciones documentadas en español.
- Los módulos de `src/` se importan de forma **plana** (`from theme import ...`),
  no como paquete. El orquestador agrega `src/` a `sys.path`.

### Por qué existe `src/theme.py` (no estaba en la estructura pedida)

`PLOTLY_LAYOUT`, la paleta de colores (`C_OK`, `C_ERR`, …) y el helper `apl()`
son globales que usan **todos** los módulos de gráficas. Se aislaron en
`theme.py` para no duplicarlos 8 veces. Es **solo presentación, cero lógica
financiera**, copiado tal cual del legacy.

## Mapeo módulo legacy → archivo

El notebook legacy era monolítico: toda la lógica vivía en una sola celda dentro
del callback `ejecutar_analisis(b)` (~1500 líneas). Se descompuso así:

| Legacy (módulo) | Funciones | Archivo nuevo |
|---|---|---|
| A Descarga + estadísticos | `descargar_precios`, `estadisticos_base` | `fetch.py` |
| B/C/D Cov, optimizadores, frontera | `cov_ledoit_wolf`, `optimizar`, `black_litterman`, `simular_frontera`, `fig_heatmap_corr`, `fig_frontera` | `optimization.py` |
| E/G/H Riesgo, drawdown, rolling | `calcular_riesgo_completo`, `fig_var_distribucion`, `calcular_drawdown`, `fig_drawdown`, `rolling_metrics`, `fig_rolling` | `risk.py` |
| F Stress testing | `PERIODOS_STRESS`, `stress_testing`, `fig_stress` | `stress.py` |
| I Factores Fama-French | `descargar_factores_ff`, `regresion_factores`, `fig_factor_betas` | `factors.py` |
| J Régimen de mercado | `detectar_regimenes`, `fig_regimenes` | `regime.py` |
| K Monte Carlo forward | `monte_carlo_forward`, `fig_monte_carlo` | `projection.py` |
| L Rebalanceo | `calcular_rebalanceo`, `fig_rebalanceo` | `rebalance.py` |
| M Análisis técnico | `calcular_rsi`, `calcular_macd`, `calcular_estocastico`, `detectar_señales`, `fig_tecnica`, `fig_base100` | `technical.py` |
| O/P Excel + PDF | `_t`, `PortfolioPDF`, `exportar_excel`, `exportar_pdf` | `export.py` |
| N Dashboard HTML | — | **NO portado** (fuera de alcance esta fase) |

La lógica y las fórmulas se preservaron **idénticas** al legacy. Lo único que
cambió fue lo específico de Colab (ver abajo).

## Correcciones conocidas (NO romper)

### Heredadas del legacy (correcciones ya resueltas — mantener intactas)
- **`export._t()`**: sanitiza texto a latin-1 para fpdf2 (acentos, guion largo
  `—`, μ, σ, ≥, ≤). fpdf2 con fuentes core no soporta Unicode arbitrario.
  Todo texto que entra al PDF debe pasar por `_t()`.
- **Tema Plotly**: `pio.templates.default='plotly_dark'` y el dict
  `PLOTLY_LAYOUT` viven en `theme.py`. Mantener.

### Correcciones aplicadas durante el port (bugs preexistentes del legacy)
Se encontraron 3 bugs latentes en el notebook legacy al correrlo contra Plotly
6.x / pandas 3.0 / fpdf2 2.8. Se arreglaron de forma mínima conservando la
intención visual; cada uno está marcado con `# FIX (port)` en el código:
1. **`risk.fig_drawdown`** — generaba un `fillcolor` malformado
   (`'rgba(ff7b72'`). Se reemplazó por conversión hex→rgba correcta.
2. **`technical.fig_tecnica`** — pasaba `legend` por desempaquetado de
   `PLOTLY_LAYOUT` **y** como kwarg explícito → `TypeError`. Se excluye `legend`
   del desempaquetado.
3. **`export.exportar_pdf`** — un literal con `—` no estaba envuelto en `_t()`
   (el autor olvidó su propio fix en esa línea). Se envolvió.

### Adaptaciones Colab → local
- `google.colab.files.download()` → escritura normal a `outputs/`.
- `from tqdm.notebook import tqdm` → `from tqdm.auto import tqdm` (portable).
- `DataFrame.applymap` → `DataFrame.map` (applymap fue removido en pandas 3.0).
- Los widgets de `ipywidgets` se reemplazaron por parámetros editables al inicio
  de `notebooks/analisis.py`.

## Cómo correr

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python notebooks/analisis.py          # corre el pipeline completo (datos sintéticos)
# o ejecutar por celdas # %% en el Interactive Window de VS Code
```

- `notebooks/analisis.py` tiene `USAR_DATOS_SINTETICOS = True` por defecto
  (GBM offline, sin red). Ponerlo en `False` para usar datos reales de yfinance.
- Salidas en `outputs/`: `portfolio_v3.xlsx` (18 hojas) y
  `portfolio_report_v3.pdf`. Si `kaleido` está instalado, el PDF embebe las
  gráficas; si no, se genera sin ellas (fallback automático).
- Convertir a notebook para entrega: `jupytext --to notebook notebooks/analisis.py`.

## Dependencias — notas

- `statsmodels` y `tqdm` **no estaban** en la lista original pero son requeridos
  (`factors.py` y barras de progreso). Se agregaron a `requirements.txt`.
- `PyPortfolioOpt` está listado por requerimiento del proyecto pero la lógica
  portada **no lo usa** (optimiza con `scipy.optimize` + `sklearn.LedoitWolf`).
- `google-genai` está listado para la fase siguiente; aún no se usa.
- Verificado funcionando con: Python 3.14, numpy 2.5, pandas 3.0, scipy 1.18,
  scikit-learn 1.9, statsmodels 0.14, plotly 6.8, fpdf2 2.8, openpyxl 3.1.

## Estado del proyecto

### ✅ Portado y verificado (esta fase)
- Pipeline analítico completo (módulos A–M) corriendo de punta a punta.
- Exportación Excel + PDF.
- Datos sintéticos para desarrollo/pruebas.

### ⏳ Pendiente (fases siguientes — NO implementar sin pedir)
- **Análisis fundamental**
- **Señales de swing trading**
- **Módulo de noticias** (probablemente con `google-genai`)
- **Dashboard nuevo** (el HTML del legacy, módulo N, no se portó; `dashboard/`
  está vacío a propósito)

## Estado actual

- ✅ Portados: `fetch`, `optimization`, `risk`, `stress`, `factors`, `regime`,
  `projection`, `rebalance`, `technical`, `export`
- ⏳ Pendientes fase 2: `fundamental.py`, `signals.py`, `news.py`
- ⏳ Pendientes fase 2: `dashboard/` (base HTML ya existe en legacy como
  `financial_intelligence_system_rams_swiss_visuals_v2.html`)
- 🐛 Nota: `kaleido` debe estar instalado para que el PDF incluya gráficas
  embebidas
- 🐛 Nota: el nombre real del notebook legacy es
  `markowitz_v3_institucional.ipynb`

## Convenciones

- No comparar números contra el notebook viejo: los datos son distintos
  (sintéticos vs reales).
- Si algo de la lógica del legacy no está claro, **preguntar antes de inventar**.
- No tocar el dashboard HTML ni adelantar las fases pendientes sin confirmación.
