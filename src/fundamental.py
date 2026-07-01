# %% [markdown]
# # fundamental — Análisis fundamental por activo (Fase 2)
#
# Extrae métricas fundamentales de cada ticker desde yfinance (`.info`).
# Metodología: se toma la foto de fundamentales que publica Yahoo Finance por
# activo y se organiza en un DataFrame homogéneo (índice = ticker).
#
# **Robustez:** los ETFs (VOO, VT, SLV, QQQM…) no reportan P/E, márgenes, ROE,
# etc. Cuando yfinance no devuelve un campo, se rellena con `None` sin romper el
# pipeline. Ningún ticker faltante lanza excepción.

# %%
import time

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm.auto import tqdm


# %% [markdown]
# ## Métricas objetivo
#
# Mapa columna_salida → clave de `Ticker.info` de yfinance. El orden define las
# columnas del DataFrame resultante. `dist_52w_high_%` se calcula aparte.

# %%
CAMPOS_INFO = {
    # Valuación
    'pe_trailing':      'trailingPE',
    'pe_forward':       'forwardPE',
    'price_to_book':    'priceToBook',
    'ev_ebitda':        'enterpriseToEbitda',
    # Rentabilidad
    'margen_bruto':     'grossMargins',
    'margen_operativo': 'operatingMargins',
    'margen_neto':      'profitMargins',
    'roe':              'returnOnEquity',
    'roa':              'returnOnAssets',
    # Crecimiento
    'revenue_yoy':      'revenueGrowth',
    'earnings_yoy':     'earningsGrowth',
    # Solidez
    'deuda_equity':     'debtToEquity',
    'current_ratio':    'currentRatio',
    'beta':             'beta',
    # Precio
    'high_52w':         'fiftyTwoWeekHigh',
    'low_52w':          'fiftyTwoWeekLow',
    'precio_actual':    'currentPrice',
}

COLUMNAS = list(CAMPOS_INFO.keys()) + ['dist_52w_high_pct']


# %% [markdown]
# ## Extracción por ticker

# %%
def _fila_fundamental(ticker):
    """Extrae la fila de métricas de un ticker. Nunca lanza: si algo falla o un
    campo no existe, deja `None` en ese campo. Retorna dict columna→valor."""
    fila = {c: None for c in COLUMNAS}
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as e:
        print(f'  ⚠ {ticker}: info no disponible ({e})')
        return fila

    for col, clave in CAMPOS_INFO.items():
        val = info.get(clave)
        # yfinance a veces devuelve strings vacíos o 'Infinity'; normalizar a None
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            fila[col] = None
        else:
            fila[col] = val

    # 'currentPrice' puede faltar en algunos activos → fallback a regularMarketPrice
    if fila['precio_actual'] is None:
        fila['precio_actual'] = info.get('regularMarketPrice')

    # Distancia al máximo de 52 semanas (%): (precio - high) / high * 100
    precio, high = fila['precio_actual'], fila['high_52w']
    if precio is not None and high not in (None, 0):
        fila['dist_52w_high_pct'] = (precio - high) / high * 100.0
    return fila


# %% [markdown]
# ## Función principal

# %%
def obtener_fundamentales(tickers, periodo="1y"):
    """Obtiene métricas fundamentales de una lista de tickers vía yfinance.

    Parámetros
    ----------
    tickers : lista de símbolos (los 15 del proyecto).
    periodo : reservado por compatibilidad de firma (las métricas de `.info`
              son una foto puntual; no requieren ventana temporal).

    Retorna
    -------
    dict con clave 'metricas' → DataFrame (índice = ticker, columnas = métricas).
    Los campos ausentes quedan como `None`. Nunca lanza por un ticker faltante.
    """
    filas = {}
    for tk in tqdm(tickers, desc='Fundamentales'):
        filas[tk] = _fila_fundamental(tk)
    df = pd.DataFrame.from_dict(filas, orient='index', columns=COLUMNAS)
    df.index.name = 'ticker'
    return {'metricas': df}


# %%
if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from fetch import cargar_portafolio_csv
    tickers, _ = cargar_portafolio_csv(
        os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic_portfolio.csv'))
    res = obtener_fundamentales(tickers[:3])
    print(res['metricas'].T)
