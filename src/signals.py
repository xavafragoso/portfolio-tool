# %% [markdown]
# # signals — Señales de swing trading (Triple-Barrier) (Fase 2)
#
# Metodología: **Triple-Barrier simplificado** (Marcos López de Prado,
# *Advances in Financial Machine Learning*) — **sin meta-labeling ni ML**.
#
# 1. **Entrada** cuando se cumplen simultáneamente: cruce alcista SMA20/SMA50,
#    RSI14 en zona neutral (40–70) e histograma MACD positivo.
# 2. Al entrar se fijan **3 barreras**: take-profit (`+k_tp·ATR`), stop-loss
#    (`−k_sl·ATR`) y **barrera vertical** (`N` días hábiles).
# 3. Se recorre el precio hacia adelante: gana la barrera que se toca primero
#    (TP / SL); si vence el plazo sin tocarse, resultado `'tiempo'`.
# 4. Señales generadas en los últimos `N` días aún sin resolver → `'abierta'`.
#
# Los indicadores (SMA20, SMA50, RSI14, MACD, ATR14) se calculan aquí mismo
# para evitar dependencia circular con `technical.py`.
#
# > **Nota de datos:** el proyecto usa series **solo-cierre** (sin OHLC). El
# > ATR14 se aproxima con el *true range* close-to-close: `|close_t − close_{t-1}|`
# > suavizado (Wilder). Es una aproximación coherente con el resto del pipeline.

# %%
import numpy as np
import pandas as pd


# %% [markdown]
# ## Indicadores técnicos (cálculo local, mínimo)

# %%
def _sma(c, n):
    """Media móvil simple de `n` periodos."""
    return c.rolling(n).mean()


def _rsi(c, p=14):
    """RSI de Wilder (suavizado exponencial alpha=1/p)."""
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1 / p, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / p, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def _macd(c, f=12, s=26, sig=9):
    """MACD: retorna (línea MACD, línea señal, histograma)."""
    m = c.ewm(span=f, adjust=False).mean() - c.ewm(span=s, adjust=False).mean()
    sg = m.ewm(span=sig, adjust=False).mean()
    return m, sg, m - sg


def _atr(c, p=14):
    """ATR14 aproximado a partir de precios solo-cierre.

    True range proxy = |close_t - close_{t-1}|, suavizado con Wilder (ewm).
    """
    tr = c.diff().abs()
    return tr.ewm(alpha=1 / p, adjust=False).mean()


# %% [markdown]
# ## Generación de señales (una serie de precios)

# %%
def _senales_ticker(ticker, c, k_tp, k_sl, N):
    """Genera y resuelve señales triple-barrier para la serie de cierres `c`
    de un ticker. Retorna lista de dicts."""
    c = c.dropna()
    if len(c) < 55:  # mínimo para SMA50 + margen
        return []

    sma20, sma50 = _sma(c, 20), _sma(c, 50)
    rsi = _rsi(c, 14)
    macd, signal, _hist = _macd(c)
    atr = _atr(c, 14)

    fechas = c.index
    n = len(c)
    ultimo = n - 1
    out = []

    # Recorremos desde que SMA50 es válida (i>=50) y hay barra previa (i>=1)
    for i in range(50, n):
        cruce_alcista = (sma20.iloc[i] > sma50.iloc[i]) and (sma20.iloc[i - 1] <= sma50.iloc[i - 1])
        rsi_ok  = 40 <= rsi.iloc[i] <= 70
        macd_ok = macd.iloc[i] > signal.iloc[i]
        if not (cruce_alcista and rsi_ok and macd_ok):
            continue

        precio_entrada = float(c.iloc[i])
        atr_i = float(atr.iloc[i])
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue  # sin ATR válido no se pueden fijar barreras

        take_profit = precio_entrada + k_tp * atr_i
        stop_loss   = precio_entrada - k_sl * atr_i
        idx_limite  = i + N
        fecha_limite = str(fechas[idx_limite].date()) if idx_limite < n else None
        razon = (f'Cruce alcista SMA20>SMA50; RSI14={rsi.iloc[i]:.1f} (40-70); '
                 f'MACD hist>0')

        # Resolver recorriendo hacia adelante hasta min(i+N, último)
        resultado = fecha_res = precio_res = None
        fin_scan = min(idx_limite, ultimo)
        for j in range(i + 1, fin_scan + 1):
            precio_j = float(c.iloc[j])
            if precio_j >= take_profit:
                resultado, fecha_res, precio_res = 'TP', fechas[j], precio_j; break
            if precio_j <= stop_loss:
                resultado, fecha_res, precio_res = 'SL', fechas[j], precio_j; break

        if resultado is None:
            if idx_limite <= ultimo:
                # Venció la barrera vertical sin tocar TP/SL
                resultado = 'tiempo'
                fecha_res, precio_res = fechas[idx_limite], float(c.iloc[idx_limite])
            else:
                # No hay suficientes barras futuras → señal abierta
                resultado = 'abierta'

        retorno_pct = ((precio_res - precio_entrada) / precio_entrada * 100.0
                       if precio_res is not None else None)

        out.append({
            'ticker':            ticker,
            'fecha_entrada':     str(fechas[i].date()),
            'precio_entrada':    round(precio_entrada, 4),
            'take_profit':       round(take_profit, 4),
            'stop_loss':         round(stop_loss, 4),
            'fecha_limite':      fecha_limite,
            'razon_entrada':     razon,
            'resultado':         resultado,
            'fecha_resolucion':  str(fecha_res.date()) if fecha_res is not None else None,
            'precio_resolucion': round(precio_res, 4) if precio_res is not None else None,
            'retorno_pct':       round(retorno_pct, 4) if retorno_pct is not None else None,
        })
    return out


# %% [markdown]
# ## Función principal

# %%
def generar_senales(precios_df, k_tp=2.0, k_sl=1.5, N=20):
    """Genera señales de swing trading (triple-barrier) para todos los activos.

    Parámetros
    ----------
    precios_df : DataFrame de precios de cierre (índice = fechas, columnas = tickers).
    k_tp, k_sl : múltiplos de ATR para take-profit y stop-loss.
    N          : horizonte de la barrera vertical (días hábiles).

    Retorna
    -------
    lista de dicts, una entrada por señal detectada (ver `_senales_ticker`).
    """
    senales = []
    for tk in precios_df.columns:
        senales.extend(_senales_ticker(tk, precios_df[tk], k_tp, k_sl, N))
    return senales


# %%
if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from fetch import generar_precios_sinteticos, cargar_portafolio_csv
    tickers, _ = cargar_portafolio_csv(
        os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic_portfolio.csv'))
    precios = generar_precios_sinteticos(tickers, n_dias=1260, semilla=42)
    sen = generar_senales(precios)
    print(f'Señales generadas: {len(sen)}')
    if sen:
        df = pd.DataFrame(sen)
        print(df['resultado'].value_counts().to_string())
        print('\nEjemplo:'); print(pd.Series(sen[0]).to_string())
