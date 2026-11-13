"""
CineMatch — Sistema de Recomendación Híbrido (MovieLens)
Proyecto final de Minería de Datos — Punto 3

Combina 4 modelos entrenados en Colab:
  • Filtrado colaborativo item-based (similitud coseno, k vecinos)
  • Matrix Factorization (SVD truncada)
  • Basado en contenido (TF-IDF de géneros + tags)
  • Neural Collaborative Filtering (NeuMF), inferencia en numpy
Ejecutar:  streamlit run app.py
"""
import os, json, pickle, re
import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

BASE = os.path.dirname(os.path.abspath(__file__))
ARTS = os.environ.get('ARTEFACTOS', os.path.join(BASE, 'artefactos_app'))
# ID del archivo artefactos_app.zip en Google Drive (compartido como "Cualquier persona con el enlace")
DRIVE_ZIP_ID = '1uxpG8tHPcJgnfVW5K2D45p4VOwdKj0nU'

st.set_page_config(page_title='CineMatch · Recomendador Híbrido', page_icon='🎬', layout='wide')


@st.cache_resource(show_spinner='Descargando modelos entrenados (≈31 MB, solo la primera vez)…')
def asegurar_artefactos():
    """En Streamlit Cloud los artefactos no están en el repo: se descargan de Drive una sola vez.
    cache_resource hace que, si varias sesiones abren la app a la vez, solo una descargue y las demás esperen."""
    if os.path.exists(os.path.join(ARTS, 'config.json')):
        return True
    import zipfile, tempfile, gdown
    fd, destino = tempfile.mkstemp(suffix='.zip')   # archivo temporal único (evita choques entre sesiones)
    os.close(fd)
    try:
        gdown.download(id=DRIVE_ZIP_ID, output=destino, quiet=True)
        if not zipfile.is_zipfile(destino):
            raise RuntimeError('No se pudo descargar artefactos_app.zip desde Google Drive. '
                               'Revisa que el archivo esté compartido como «Cualquier persona con el enlace».')
        with zipfile.ZipFile(destino) as z:
            z.extractall(os.path.dirname(ARTS))
    finally:
        if os.path.exists(destino):
            os.remove(destino)
    return True


asegurar_artefactos()

st.markdown("""
<style>
.card {border:1px solid rgba(128,128,128,.25); border-radius:12px; padding:12px 14px; margin-bottom:10px;
       background:rgba(128,128,128,.06); min-height:170px}
.card h4 {margin:0 0 4px 0; font-size:1rem; line-height:1.25}
.rank {display:inline-block; background:#e50914; color:white; border-radius:6px; padding:0 7px;
       font-weight:700; font-size:.8rem; margin-right:6px}
.genre {display:inline-block; font-size:.72rem; border-radius:10px; padding:1px 8px; margin:2px 3px 2px 0;
        background:rgba(229,9,20,.12); color:inherit}
.why {font-size:.8rem; opacity:.8; margin-top:6px}
.score {font-size:.8rem; opacity:.75}
a.lnk {font-size:.78rem; margin-right:8px}
</style>
""", unsafe_allow_html=True)

NOMBRES = {'cf': 'CF item-based', 'mf': 'Matrix Factorization (SVD)',
           'contenido': 'Basado en contenido', 'ncf': 'NeuMF (NCF)', 'hibrido': 'HÍBRIDO'}


# ---------------------------------------------------------------------
# Carga de artefactos
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner='Cargando modelos…')
def cargar():
    a = {}
    a['movies'] = pd.read_parquet(f'{ARTS}/movies.parquet').sort_values('movie_idx').reset_index(drop=True)
    a['movies']['year'] = pd.to_numeric(a['movies']['title'].str.extract(r'\((\d{4})\)\s*$')[0], errors='coerce')
    a['n_movies'] = len(a['movies'])
    with open(f'{ARTS}/config.json') as fh:
        a['config'] = json.load(fh)
    with open(f'{ARTS}/mapeos.pkl', 'rb') as fh:
        a['mapeos'] = pickle.load(fh)

    v = np.load(f'{ARTS}/cf_vecinos.npz')
    n, k = v['idx'].shape
    a['S'] = sp.csr_matrix((v['sim'].ravel(), (np.repeat(np.arange(n), k), v['idx'].ravel())), shape=(n, n))
    a['V'] = np.load(f'{ARTS}/mf_V.npy').astype(np.float32)
    a['C'] = sp.load_npz(f'{ARTS}/contenido_C.npz').tocsr().astype(np.float32)
    a['R'] = sp.load_npz(f'{ARTS}/R_centrada.npz').tocsr()
    a['seen'] = sp.load_npz(f'{ARTS}/seen.npz').tocsr()
    a['media_user'] = np.load(f'{ARTS}/media_user.npy')
    a['popularidad'] = np.load(f'{ARTS}/popularidad.npy')
    a['ncf'] = {k: v for k, v in np.load(f'{ARTS}/ncf_pesos.npz').items()}

    busq = pd.read_csv(f'{ARTS}/busqueda_pesos_hibrido.csv')
    a['busqueda'] = busq
    comp = ['cf', 'mf', 'contenido', 'ncf']
    a['pesos'] = {c: float(busq.iloc[0][c]) for c in comp}
    sin_ncf = busq[busq['ncf'] == 0].iloc[0]          # mejores pesos cuando NO hay NCF (usuario nuevo)
    a['pesos_nuevo'] = {c: float(sin_ncf[c]) for c in comp}

    p = f'{ARTS}/resultados_modelos.csv'
    a['resultados'] = pd.read_csv(p) if os.path.exists(p) else None
    p = f'{ARTS}/historial_ncf.csv'
    a['historial'] = pd.read_csv(p) if os.path.exists(p) else None
    return a


try:
    A = cargar()
except Exception as e:
    st.error(f"No se pudieron cargar los artefactos desde `{ARTS}`.\n\n{e}")
    st.stop()

MOVIES, N = A['movies'], A['n_movies']
TITULOS = MOVIES['title'].tolist()
GENEROS = sorted({g for gs in MOVIES['genres'] for g in gs.split('|') if g != '(no genres listed)'})


# ---------------------------------------------------------------------
# Modelos (puntaje de todas las películas para un vector de ratings centrados)
# ---------------------------------------------------------------------
def s_cf(r):          # r: (1, n_movies) sparse centrado
    return np.asarray((r @ A['S']).todense()).ravel()

def s_mf(r):
    return np.asarray(r @ A['V']).ravel() @ A['V'].T

def s_contenido(r):
    return np.asarray(((r @ A['C']) @ A['C'].T).todense()).ravel()

def s_ncf(u):
    w = A['ncf']
    gmf = w['u_gmf.weight'][u] * w['i_gmf.weight']
    h = np.hstack([np.broadcast_to(w['u_mlp.weight'][u], w['i_mlp.weight'].shape), w['i_mlp.weight']])
    capas = sorted({int(k.split('.')[1]) for k in w if k.startswith('mlp.') and k.endswith('.weight')})
    for c in capas:
        h = np.maximum(h @ w[f'mlp.{c}.weight'].T + w[f'mlp.{c}.bias'], 0)
    logit = np.hstack([gmf, h]) @ w['salida.weight'].T + w['salida.bias']
    return 1 / (1 + np.exp(-logit.ravel()))

def z(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def puntajes(r, u=None):
    """Devuelve dict con los puntajes (z-score) de cada componente."""
    out = {'cf': z(s_cf(r)), 'mf': z(s_mf(r)), 'contenido': z(s_contenido(r))}
    if u is not None:
        out['ncf'] = z(s_ncf(u))
    return out


def recomendar(comp, pesos, modelo, vistas, n, generos=None, rango=None):
    if modelo == 'hibrido':
        total = sum(pesos[c] * comp[c] for c in comp if pesos.get(c, 0) > 0)
    else:
        total = comp[modelo].copy()
    total = np.array(total, dtype=np.float64)
    total[list(vistas)] = -np.inf
    if generos:
        ok = MOVIES['genres'].apply(lambda g: any(x in g.split('|') for x in generos)).values
        total[~ok] = -np.inf
    if rango:
        y = MOVIES['year'].values
        total[~((y >= rango[0]) & (y <= rango[1]))] = -np.inf
    top = np.argsort(-total)[:n]
    return [i for i in top if np.isfinite(total[i])], total


def explicar(m, r_dict):
    """Película calificada por el usuario que más aporta a la recomendación m (vía vecinos CF y contenido)."""
    idx = np.array([j for j, v in r_dict.items() if v > 0], dtype=int)
    if len(idx) == 0:
        return None
    rv = np.array([r_dict[j] for j in idx])
    sim = A['S'][idx, m].toarray().ravel() + (A['C'][idx] @ A['C'][m].T).toarray().ravel()
    aporte = rv * sim
    k = int(np.argmax(aporte))
    return int(idx[k]) if aporte[k] > 0 else None


# ---------------------------------------------------------------------
# Presentación
# ---------------------------------------------------------------------
def links(row):
    out = ''
    if pd.notna(row.get('imdbId')):
        out += f"<a class='lnk' href='https://www.imdb.com/title/tt{int(row['imdbId']):07d}/' target='_blank'>IMDb</a>"
    if pd.notna(row.get('tmdbId')):
        out += f"<a class='lnk' href='https://www.themoviedb.org/movie/{int(row['tmdbId'])}' target='_blank'>TMDB</a>"
    return out


def tarjetas(recs, total, r_dict=None, cols=4):
    filas = [recs[i:i + cols] for i in range(0, len(recs), cols)]
    pos = 0
    for fila in filas:
        cs = st.columns(cols)
        for c, m in zip(cs, fila):
            pos += 1
            row = MOVIES.iloc[m]
            gen = ''.join(f"<span class='genre'>{g}</span>" for g in row['genres'].split('|')[:4])
            why = ''
            if r_dict:
                e = explicar(m, r_dict)
                if e is not None:
                    why = f"<div class='why'>💡 Porque te gustó <b>{MOVIES.iloc[e]['title']}</b></div>"
            c.markdown(f"<div class='card'><h4><span class='rank'>{pos}</span>{row['title']}</h4>{gen}"
                       f"<div class='score'>Puntaje: {total[m]:.2f}</div>{why}<div>{links(row)}</div></div>",
                       unsafe_allow_html=True)


def grafico_aportes(recs, comp, pesos):
    filas = []
    for m in recs:
        for c in comp:
            if pesos.get(c, 0) > 0:
                filas.append({'Película': MOVIES.iloc[m]['title'][:45], 'Componente': NOMBRES[c],
                              'Aporte': pesos[c] * comp[c][m]})
    if not filas:
        return
    df = pd.DataFrame(filas)
    fig = px.bar(df, x='Aporte', y='Película', color='Componente', orientation='h',
                 title='¿Qué modelo empujó cada recomendación? (aporte ponderado, z-score)')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=120 + 32 * len(recs),
                      legend_title_text='', margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width='stretch')


def grafico_generos(ids, titulo):
    g = pd.Series([x for m in ids for x in MOVIES.iloc[m]['genres'].split('|')]).value_counts().head(10)
    fig = px.bar(x=g.values, y=g.index, orientation='h', title=titulo, labels={'x': 'Películas', 'y': ''})
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=320, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def filtros(key):
    with st.expander('⚙️ Opciones', expanded=False):
        c1, c2, c3 = st.columns([1.2, 1, 1.4])
        modelo = c1.selectbox('Modelo', list(NOMBRES), index=4, format_func=NOMBRES.get, key=f'mod_{key}')
        n = c2.slider('Cantidad', 5, 30, 12, key=f'n_{key}')
        ys = MOVIES['year'].dropna()
        rango = c3.slider('Años', int(ys.min()), int(ys.max()), (int(ys.min()), int(ys.max())), key=f'y_{key}')
        generos = st.multiselect('Filtrar por género', GENEROS, key=f'g_{key}')
    return modelo, n, generos, rango


# ---------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------
st.title('🎬 CineMatch')
st.caption('Sistema de recomendación híbrido · Filtrado colaborativo + Matrix Factorization + '
           'Contenido + Neural Collaborative Filtering · MovieLens')

with st.sidebar:
    st.header('Sobre el sistema')
    st.metric('Películas en el catálogo', f"{N:,}")
    st.metric('Usuarios de entrenamiento', f"{A['R'].shape[0]:,}")
    st.markdown('**Pesos del híbrido**')
    CORTOS = {'cf': 'CF', 'mf': 'SVD', 'contenido': 'Contenido', 'ncf': 'NeuMF'}
    pesos_df = pd.DataFrame({'Modelo': [CORTOS[c] for c in CORTOS],
                             'Existente': [A['pesos'][c] for c in CORTOS],
                             'Nuevo': [A['pesos_nuevo'][c] for c in CORTOS]})
    st.dataframe(pesos_df, hide_index=True, width='stretch',
                 column_config={'Existente': st.column_config.NumberColumn(format='%.2f', width='small'),
                                'Nuevo': st.column_config.NumberColumn(format='%.2f', width='small')})
    st.caption('Los pesos se ajustaron con búsqueda en grilla sobre usuarios de validación (Recall@10). '
               'Un usuario nuevo no tiene embedding NeuMF, por eso usa la mejor combinación sin NCF.')

tab_nuevo, tab_exist, tab_sim, tab_eval = st.tabs(
    ['✨ Crea tu perfil', '👤 Usuario existente', '🔎 Películas similares', '📊 Evaluación de modelos'])

# --- 1. Usuario nuevo -------------------------------------------------
with tab_nuevo:
    st.subheader('Califica algunas películas y te recomendamos otras')
    populares = np.argsort(-A['popularidad'])[:300]
    sugeridas = [TITULOS[i] for i in populares]
    elegidas = st.multiselect('Busca y elige películas que hayas visto (mínimo 3):', TITULOS,
                              default=sugeridas[:0], placeholder='Escribe un título… ej. Toy Story, Matrix',
                              key='elegidas')
    if not elegidas:
        st.info('Sugerencias populares: ' + ' · '.join(sugeridas[:12]))
    r_dict_raw = {}
    if elegidas:
        cols = st.columns(3)
        for i, t in enumerate(elegidas):
            r_dict_raw[TITULOS.index(t)] = cols[i % 3].slider(t, 0.5, 5.0, 4.0, 0.5, key=f'r_{t}')
    modelo, n, generos, rango = filtros('nuevo')
    if modelo == 'ncf':
        st.warning('NeuMF necesita un usuario visto en entrenamiento. Para perfiles nuevos usa los otros modelos.')
    elif len(r_dict_raw) >= 3:
        vals = np.array(list(r_dict_raw.values()))
        media = vals.mean() if vals.std() > 0 else 3.0     # si calificó todo igual, centramos en 3
        r_dict = {k: v - media for k, v in r_dict_raw.items()}
        r = sp.csr_matrix((list(r_dict.values()), ([0] * len(r_dict), list(r_dict.keys()))), shape=(1, N))
        comp = puntajes(r)
        recs, total = recomendar(comp, A['pesos_nuevo'], modelo, r_dict.keys(), n, generos, rango)
        st.markdown(f'### Recomendado para ti · {NOMBRES[modelo]}')
        tarjetas(recs, total, r_dict)
        if modelo == 'hibrido':
            grafico_aportes(recs, comp, A['pesos_nuevo'])
    elif elegidas:
        st.info('Elige al menos 3 películas.')

# --- 2. Usuario existente ---------------------------------------------
with tab_exist:
    st.subheader('Recomendaciones para un usuario de MovieLens')
    u2i = A['mapeos']['user2idx']
    ids = sorted(u2i)
    c1, c2 = st.columns([3, 1])
    if c2.button('🎲 Usuario aleatorio'):
        st.session_state['uid'] = int(np.random.choice(ids))
    uid = c1.selectbox('userId (MovieLens)', ids, index=ids.index(st.session_state.get('uid', ids[0])))
    u = u2i[uid]
    fila = A['R'][u]
    vistas = set(A['seen'][u].indices)
    r_dict = dict(zip(fila.indices, fila.data))
    hist = pd.DataFrame({'movie_idx': fila.indices, 'rating': fila.data + A['media_user'][u]})
    hist = hist.merge(MOVIES[['movie_idx', 'title', 'genres']], on='movie_idx').sort_values('rating', ascending=False)

    m1, m2, m3 = st.columns(3)
    m1.metric('Películas calificadas', len(hist))
    m2.metric('Rating promedio', f"{A['media_user'][u]:.2f}")
    m3.metric('Género favorito', pd.Series([g for gs in hist[hist.rating >= 4]['genres'] for g in gs.split('|')])
              .value_counts().index[0] if (hist.rating >= 4).any() else '—')

    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown('**Historial (lo mejor calificado)**')
        st.dataframe(hist[['title', 'genres', 'rating']].head(15).rename(
            columns={'title': 'Película', 'genres': 'Géneros', 'rating': 'Rating'}),
            hide_index=True, width='stretch', height=320)
    with c2:
        st.plotly_chart(grafico_generos(hist[hist.rating >= 4]['movie_idx'].tolist(),
                                        'Géneros que más le gustan'), width='stretch')

    modelo, n, generos, rango = filtros('exist')
    comp = puntajes(sp.csr_matrix(fila), u)
    recs, total = recomendar(comp, A['pesos'], modelo, vistas, n, generos, rango)
    st.markdown(f'### Top {len(recs)} · {NOMBRES[modelo]}')
    tarjetas(recs, total, r_dict)
    if modelo == 'hibrido':
        grafico_aportes(recs, comp, A['pesos'])

    with st.expander('Comparar los 5 modelos lado a lado'):
        cols = st.columns(5)
        for c, mod in zip(cols, NOMBRES):
            rr, _ = recomendar(comp, A['pesos'], mod, vistas, 10)
            c.markdown(f'**{NOMBRES[mod]}**')
            c.markdown('\n'.join(f'{i+1}. {MOVIES.iloc[m]["title"]}' for i, m in enumerate(rr)))

# --- 3. Películas similares -------------------------------------------
with tab_sim:
    st.subheader('Encuentra películas parecidas')
    t = st.selectbox('Película', TITULOS, index=0)
    m = TITULOS.index(t)
    c1, c2 = st.columns(2)
    sim_cf = A['S'][m].toarray().ravel()
    sim_ct = (A['C'] @ A['C'][m].T).toarray().ravel(); sim_ct[m] = -1
    sim_mf = A['V'] @ A['V'][m] / (np.linalg.norm(A['V'], axis=1) * np.linalg.norm(A['V'][m]) + 1e-8); sim_mf[m] = -1
    for col, nombre, s in [(c1, 'Quienes la vieron también vieron (CF)', sim_cf),
                           (c2, 'Parecidas por contenido (géneros + tags)', sim_ct)]:
        col.markdown(f'**{nombre}**')
        top = np.argsort(-s)[:10]
        col.dataframe(pd.DataFrame({'Película': [TITULOS[i] for i in top],
                                    'Géneros': [MOVIES.iloc[i]['genres'] for i in top],
                                    'Similitud': [round(float(s[i]), 3) for i in top]}),
                      hide_index=True, width='stretch')
    st.markdown('**Cercanas en el espacio latente (SVD)**')
    top = np.argsort(-sim_mf)[:10]
    st.write(' · '.join(TITULOS[i] for i in top))

# --- 4. Evaluación -----------------------------------------------------
with tab_eval:
    st.subheader('Evaluación offline (3.000 usuarios de prueba, split temporal 80/20)')
    st.caption('Relevante = película que el usuario vio después (test) y calificó con 4 o más.')
    res = A['resultados']
    if res is not None:
        res = res.set_index('Modelo')
        st.dataframe(res.style.format('{:.4f}').highlight_max(axis=0, color='#c6efce'), width='stretch')
        c1, c2 = st.columns(2)
        d = res[['Precision@10', 'Recall@10']].reset_index().melt('Modelo', var_name='Métrica', value_name='Valor')
        c1.plotly_chart(px.bar(d, x='Valor', y='Modelo', color='Métrica', barmode='group', orientation='h',
                               title='Precision@10 y Recall@10'), width='stretch')
        fig = go.Figure()
        for mod in res.index:
            fig.add_trace(go.Scatter(x=[5, 10, 20], y=[res.loc[mod, f'Recall@{k}'] for k in (5, 10, 20)],
                                     mode='lines+markers', name=mod))
        fig.update_layout(title='Recall@K', xaxis_title='K', yaxis_title='Recall')
        c2.plotly_chart(fig, width='stretch')
    if A['historial'] is not None:
        h = A['historial']
        st.plotly_chart(px.line(h, x='epoca', y='recall@10_val', markers=True,
                                title='Entrenamiento NeuMF: Recall@10 en validación por época'),
                        width='stretch')
    st.markdown('**Búsqueda de pesos del híbrido (top 10)**')
    st.dataframe(A['busqueda'].head(10), hide_index=True, width='stretch')
