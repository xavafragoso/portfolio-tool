# Financial Intelligence System — Portfolio Tool

Sistema profesional de análisis de portafolios de inversión construido con Python e IA, desarrollado como **Proyecto Integrador del Módulo 8** del *Diplomado de Inteligencia Artificial Aplicada al Análisis Financiero* — Tecnológico de Monterrey.

---

## Demo

Para correr el dashboard localmente:

```bash
git clone https://github.com/xavafragoso/portfolio-tool
cd portfolio-tool/dashboard
python3 -m http.server 8000
```

Luego abrir **http://localhost:8000** en el navegador.

> El dashboard se sirve con un servidor local porque lee `data.json` mediante `fetch()`. Si se abre el archivo directo (`file://`), entra en modo demo con datos de respaldo.

---

## ¿Qué hace este sistema?

Responde una sola pregunta central: **"¿Qué hago con mi dinero esta semana?"** — integrando en un único dashboard:

- **Análisis cuantitativo de portafolio** (Markowitz, Black-Litterman, Risk Parity, CVaR/VaR)
- **Análisis fundamental por ticker** (vía Yahoo Finance)
- **Análisis técnico con señales de swing trading** (metodología *triple-barrier* de López de Prado)
- **Resumen de noticias clasificadas por IA** (Google Gemini)
- **Stress testing** en 7 crisis históricas
- **Proyecciones Monte Carlo** a 1 / 3 / 5 años
- **Plan de rebalanceo** con costos de transacción

---

## Arquitectura

Flujo de datos de extremo a extremo:

```
Yahoo Finance  →  src/ (Python)  →  data.json  →  dashboard/index.html
   (datos)         (pipeline)       (contrato)      (visualización)
```

Módulos del pipeline en `src/`:

| Módulo | Responsabilidad |
|---|---|
| `fetch` | Descarga de precios y estadísticos por activo |
| `optimization` | Covarianza Ledoit-Wolf, Markowitz, Black-Litterman, Risk Parity, frontera eficiente |
| `risk` | VaR, CVaR, drawdown, Calmar, rolling Sharpe |
| `stress` | Stress testing en crisis históricas |
| `factors` | Regresión de factores Fama-French (proxy vía ETFs) |
| `regime` | Detección de régimen de mercado con K-Means |
| `projection` | Proyección Monte Carlo (movimiento browniano geométrico) |
| `rebalance` | Trade list de rebalanceo con costos |
| `technical` | Indicadores técnicos (SMA / RSI / MACD / Estocástico) |
| `fundamental` | Métricas fundamentales por ticker |
| `signals` | Señales de swing trading (triple-barrier) |
| `news` | Descarga y clasificación de noticias con Gemini |
| `export` | Exportación a Excel, PDF y `data.json` |

---

## Stack tecnológico

- **Python 3.10+** con `yfinance`, `scipy`, `scikit-learn`, `statsmodels`, `plotly`, `fpdf2`, `openpyxl` (entre otras)
- **Google Gemini API** (`gemini-2.5-flash`) para el análisis de noticias
- **Dashboard**: HTML / CSS / JavaScript con Plotly — *single file*, sin framework
- **Control de versiones**: Git + GitHub

> **Nota sobre la optimización:** la optimización de portafolios (Markowitz, Mínima Varianza, Risk Parity y Black-Litterman) está implementada directamente con `scipy.optimize` (SLSQP) y `scikit-learn` (estimador de covarianza *Ledoit-Wolf*). `PyPortfolioOpt` figura como dependencia declarada en `requirements.txt` para comparación y uso futuro, pero **no** ejecuta la optimización actual.

---

## Metodologías implementadas

- **Optimización de portafolios:** Markowitz (Máximo Sharpe), Mínima Varianza, Risk Parity y Black-Litterman
- **Riesgo:** VaR histórico / paramétrico / Monte Carlo, CVaR (Expected Shortfall), Drawdown, Calmar y Rolling Sharpe
- **Machine Learning:** K-Means para detección de régimen de mercado (Bull / Bear / Lateral)
- **Análisis de factores:** regresión OLS del portafolio contra un **proxy de 4 factores** estilo Fama-French/Carhart
- **Señales:** *Triple-barrier method* (López de Prado, *Advances in Financial Machine Learning*)
- **NLP:** *Prompt engineering* estructurado con ejemplos *few-shot* para clasificación de sentimiento

> **Nota sobre los factores:** el análisis de factores **no** usa el dataset oficial de la *Kenneth R. French Data Library*, sino un **proxy construido con retornos de ETFs públicos** (vía `statsmodels` OLS): Mercado (`SPY`−`BIL`), Tamaño/SMB (`IWM`−`SPY`), Valor/HML (`IVE`−`IWF`) y Momentum (`MTUM`−`USMV`). Son 4 factores (los 3 clásicos de Fama-French más Momentum, à la Carhart), en versión aproximada para fines educativos.

---

## Estructura del proyecto

```
portfolio-tool/
├── src/          # Módulos Python del pipeline analítico
├── dashboard/    # index.html + data.json (dashboard web)
├── notebooks/    # analisis.py — orquestador del pipeline
├── data/         # Datos sintéticos para desarrollo
├── outputs/      # Excel y PDF generados (no versionados)
├── legacy/       # Notebook original de Colab (referencia)
└── CLAUDE.md     # Documentación técnica del proyecto
```

---

## Cómo correr el pipeline completo

```bash
cd portfolio-tool
source venv/bin/activate       # o venv\Scripts\activate en Windows
python notebooks/analisis.py
```

Esto genera `dashboard/data.json` con **datos reales de Yahoo Finance**.

> **Nota:** requiere `GEMINI_API_KEY` en `.env` (ver `.env.example`) para el análisis de noticias.

---

## Configuración

Copiar `.env.example` a `.env` y completar:

```env
GEMINI_API_KEY=tu_key_de_google_ai_studio
NAV_USD=tu_capital_en_usd
```

---

## Contexto académico

- **Proyecto Integrador — Módulo 8**
- **Diplomado:** Inteligencia Artificial Aplicada al Análisis Financiero
- **Institución:** Tecnológico de Monterrey

**Herramientas de IA utilizadas en el desarrollo:**
- **Claude** — arquitectura y código
- **Google Gemini** — análisis de noticias en *runtime*
- **ChatGPT** — diseño visual

---

## Notas

- Los datos de **posiciones son sintéticos** (por privacidad).
- Los **precios de mercado son reales** vía Yahoo Finance.
- **Límite de Gemini free tier:** ~20 requests/día (una corrida completa del pipeline consume 15, uno por ticker).
