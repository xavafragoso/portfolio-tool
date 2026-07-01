# %% [markdown]
# # news — Análisis de noticias con Gemini (Fase 2)
#
# Obtiene las últimas noticias de cada ticker vía **yfinance** (gratuito, sin API
# key) y las clasifica con **Google Gemini** (`gemini-2.5-flash`) usando prompt
# engineering estructurado (rol explícito + few-shot + salida JSON estricta).
#
# La API key se lee de `.env` (`GEMINI_API_KEY`) con `python-dotenv`.
#
# **Robustez:** si un ticker no tiene noticias → lista vacía. Si Gemini falla o
# el JSON no parsea → se guarda el texto crudo en un campo `error` y se continúa
# con el siguiente ticker. Rate limit: 1 s de pausa entre tickers (tier gratuito).

# %%
import os
import json
import time

import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

MODELO = 'gemini-2.5-flash'


# %% [markdown]
# ## Extracción de noticias (yfinance)

# %%
def _extraer_noticias(ticker, max_noticias):
    """Extrae hasta `max_noticias` (título + fecha) del feed de yfinance.

    Maneja el formato nuevo (`item['content']`) y el antiguo (`item['title']`,
    `providerPublishTime` epoch). Si no hay noticias, retorna lista vacía.
    """
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception as e:
        print(f'  ⚠ {ticker}: news no disponible ({e})')
        return []

    items = []
    for it in raw[:max_noticias]:
        cont = it.get('content', it) if isinstance(it, dict) else {}
        titulo = cont.get('title') or it.get('title')
        if not titulo:
            continue
        # Fecha: ISO en formato nuevo; epoch en el antiguo
        fecha = cont.get('pubDate') or cont.get('displayTime') or it.get('providerPublishTime')
        if isinstance(fecha, (int, float)):
            fecha = time.strftime('%Y-%m-%d', time.localtime(fecha))
        items.append({'titulo': str(titulo), 'fecha': str(fecha) if fecha else 'N/D'})
    return items


# %% [markdown]
# ## Construcción del prompt (rol + few-shot + JSON estricto)

# %%
def _construir_prompt(ticker, noticias):
    """Arma el prompt estructurado para Gemini: rol de analista de portafolios,
    2 ejemplos few-shot de clasificación y esquema de salida JSON estricto."""
    ejemplos = (
        'EJEMPLOS DE CLASIFICACION:\n'
        '- Noticia: "La empresa supera expectativas de ganancias y sube su guidance anual"\n'
        '  -> sentimiento="positivo", relevancia=5\n'
        '- Noticia: "Reguladores abren investigacion antimonopolio contra la compania"\n'
        '  -> sentimiento="negativo", relevancia=4\n'
    )
    lista = '\n'.join(f'{i+1}. [{n["fecha"]}] {n["titulo"]}' for i, n in enumerate(noticias))
    schema = (
        '{"ticker": str, "titulo": str, "fecha": str, '
        '"sentimiento": "positivo"|"negativo"|"neutro", "relevancia": 1-5, '
        '"resumen": str (max 2 oraciones), "impacto_portafolio": str (max 1 oracion)}'
    )
    return (
        f'Eres un analista de portafolios de inversion. Clasifica el impacto de '
        f'cada noticia sobre el activo {ticker}.\n\n'
        f'{ejemplos}\n'
        f'NOTICIAS A ANALIZAR ({ticker}):\n{lista}\n\n'
        f'INSTRUCCIONES DE FORMATO:\n'
        f'- Responde UNICAMENTE con un arreglo JSON valido, sin markdown, sin '
        f'texto adicional, sin bloques de codigo.\n'
        f'- Un objeto por noticia, en el mismo orden, con este esquema:\n{schema}\n'
    )


# %% [markdown]
# ## Parseo defensivo de la respuesta

# %%
def _parsear_json(texto):
    """Parsea la respuesta de Gemini a lista de dicts. Tolera fences de markdown.
    Lanza ValueError si no logra parsear (el caller guarda el raw)."""
    t = texto.strip()
    if t.startswith('```'):
        # quitar fence ```json ... ```
        t = t.split('```', 2)[1] if t.count('```') >= 2 else t.strip('`')
        if t.lstrip().lower().startswith('json'):
            t = t.lstrip()[4:]
    data = json.loads(t.strip())
    if isinstance(data, dict):
        data = [data]
    return data


# %% [markdown]
# ## Función principal

# %%
def analizar_noticias(tickers, max_noticias=5):
    """Analiza noticias de cada ticker con Gemini y retorna dict {ticker: [...]}.

    Nunca detiene el pipeline: errores de red, autenticación o parseo se
    capturan por ticker y se registran en un campo `error`.
    """
    resultados = {}

    api_key = os.getenv('GEMINI_API_KEY')
    cliente = None
    if not api_key or api_key.strip() in ('', 'tu_api_key_aqui'):
        print('  ⚠ GEMINI_API_KEY ausente o placeholder — se omite el análisis LLM.')
    else:
        try:
            from google import genai
            cliente = genai.Client(api_key=api_key)
        except Exception as e:
            print(f'  ⚠ No se pudo inicializar Gemini: {e}')

    for tk in tickers:
        noticias = _extraer_noticias(tk, max_noticias)
        if not noticias:
            resultados[tk] = []
            continue
        if cliente is None:
            resultados[tk] = [{'ticker': tk, 'error': 'Cliente Gemini no disponible',
                               'titulo': n['titulo'], 'fecha': n['fecha']} for n in noticias]
            continue

        prompt = _construir_prompt(tk, noticias)
        try:
            resp = cliente.models.generate_content(model=MODELO, contents=prompt)
            raw = resp.text or ''
            resultados[tk] = _parsear_json(raw)
        except Exception as e:
            print(f'  ⚠ {tk}: fallo Gemini/parse ({type(e).__name__}: {e})')
            resultados[tk] = [{'ticker': tk, 'error': str(e),
                               'raw': locals().get('raw', '')}]
        time.sleep(1)  # rate limiting tier gratuito

    return resultados


# %%
if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from fetch import cargar_portafolio_csv
    tickers, _ = cargar_portafolio_csv(
        os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic_portfolio.csv'))
    res = analizar_noticias(tickers[:2])
    print(json.dumps(res, indent=2, ensure_ascii=False)[:2000])
