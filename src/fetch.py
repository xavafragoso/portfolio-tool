# %% [markdown]
# # fetch — Descarga de datos y estadísticos por activo
#
# Módulo A del notebook legacy. Descarga de precios vía yfinance y cálculo
# de estadísticos base por activo (rendimiento/volatilidad anualizados,
# Sharpe, skewness, kurtosis).
#
# Se añade `generar_precios_sinteticos()` (NO existe en el legacy) para poder
# correr el pipeline completo offline con datos sintéticos — el proyecto usa
# solo datos sintéticos por privacidad.

# %%
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import skew, kurtosis
from tqdm.auto import tqdm   # legacy usaba tqdm.notebook (Colab); .auto es portable


# %% [markdown]
# ## Descarga de precios (yfinance)

# %%
def descargar_precios(tickers, inicio, fin, intervalo):
    """Descarga precios de cierre desde yfinance para una lista de tickers.

    Descarta tickers con menos de 50 observaciones. Rellena NaN con
    forward/backward fill. Retorna un DataFrame (índice=Fecha, columnas=tickers).
    """
    ok, mal = {}, []
    for tk in tqdm(tickers, desc='Descargando'):
        try:
            raw = yf.download(tk, start=inicio, end=fin, interval=intervalo,
                              progress=False, auto_adjust=False)
            if raw.empty or len(raw) < 50:
                print(f'  ⚠ {tk}: insuficiente — descartado'); mal.append(tk); continue
            ok[tk] = raw['Close'].squeeze()
        except Exception as e:
            print(f'  ✗ {tk}: {e}'); mal.append(tk)
    if not ok: raise ValueError('Sin tickers válidos.')
    df = pd.DataFrame(ok); df.index = pd.to_datetime(df.index); df.index.name = 'Fecha'
    if df.isna().any().any(): df = df.ffill().bfill()
    print(f'✅ {len(df.columns)} activos | {len(df):,} obs | Excluidos: {mal or "ninguno"}')
    return df


# %% [markdown]
# ## Estadísticos base por activo

# %%
def estadisticos_base(rend, factor, rf_p):
    """Calcula estadísticos por activo a partir de rendimientos logarítmicos.

    `factor` = periodos por año (252/52/12). `rf_p` = tasa libre de riesgo
    por periodo. Retorna DataFrame con rendimiento/volatilidad anualizados,
    Sharpe, skewness, kurtosis y métricas por periodo.
    """
    mu_p = rend.mean(); std_p = rend.std(ddof=1)
    mu_a = mu_p * factor; std_a = std_p * np.sqrt(factor)
    rf_a = (1 + rf_p) ** factor - 1
    sk = rend.apply(skew); ku = rend.apply(kurtosis)
    return pd.DataFrame({'Rend_Anual': mu_a, 'Std_Anual': std_a,
                         'Sharpe': (mu_a - rf_a) / std_a,
                         'Skewness': sk, 'Kurtosis': ku,
                         'Rend_Periodo': mu_p, 'Std_Periodo': std_p})


# %% [markdown]
# ## Generador de precios sintéticos (solo desarrollo/pruebas)
#
# No forma parte del notebook legacy. Genera series de precios mediante un
# movimiento browniano geométrico (GBM) para correr el pipeline sin red y sin
# datos reales. Útil para validar que todo ejecuta de punta a punta.

# %%
def generar_precios_sinteticos(tickers, n_dias=1260, precio_inicial=100.0,
                               mu_anual=0.10, sigma_anual=0.25, semilla=42):
    """Genera precios diarios sintéticos (GBM) para una lista de tickers.

    Retorna un DataFrame con la misma forma que `descargar_precios`
    (índice=Fecha de días hábiles, columnas=tickers). Solo para desarrollo.
    """
    rng = np.random.default_rng(semilla)
    n = len(tickers)
    mu_d = mu_anual / 252
    sig_d = sigma_anual / np.sqrt(252)
    # Choques con algo de correlación común de mercado (factor latente)
    mercado = rng.normal(0, 1, (n_dias, 1))
    idiosinc = rng.normal(0, 1, (n_dias, n))
    choques = 0.5 * mercado + 0.85 * idiosinc
    log_ret = (mu_d - 0.5 * sig_d ** 2) + sig_d * choques
    precios = precio_inicial * np.exp(np.cumsum(log_ret, axis=0))
    fechas = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_dias)
    df = pd.DataFrame(precios, index=fechas, columns=tickers)
    df.index.name = 'Fecha'
    return df


# %% [markdown]
# ## Lectura del portafolio sintético (CSV de pesos)

# %%
def cargar_portafolio_csv(ruta='data/synthetic_portfolio.csv'):
    """Lee `data/synthetic_portfolio.csv` (ticker, peso_pct).

    Retorna (lista_tickers, array_pesos_decimales). Los pesos se normalizan
    a fracción (suma 1.0).
    """
    df = pd.read_csv(ruta)
    tickers = df['ticker'].tolist()
    pesos = df['peso_pct'].to_numpy(dtype=float) / 100.0
    return tickers, pesos


# %%
if __name__ == '__main__':
    tks, w = cargar_portafolio_csv()
    px_ = generar_precios_sinteticos(tks)
    rend_ = np.log(px_ / px_.shift(1)).dropna()
    print(estadisticos_base(rend_, 252, (1.05)**(1/252) - 1).round(4))
