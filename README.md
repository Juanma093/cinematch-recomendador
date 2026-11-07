# CineMatch — Sistema de Recomendación Híbrido (MovieLens)

Proyecto final de Minería de Datos — Punto 3.

App web en Streamlit que combina cuatro modelos: filtrado colaborativo item-based, Matrix Factorization (SVD), un modelo basado en contenido (TF-IDF de géneros y tags) y Neural Collaborative Filtering (NeuMF). Los modelos se entrenan en el notebook de Colab y la app solo carga los artefactos.

## Estructura
```
app.py
requirements.txt
artefactos_app/      <- se descarga sola desde Drive (artefactos_app.zip) la primera vez que corre la app
```

## Ejecutar en local
```
pip install -r requirements.txt
streamlit run app.py
```

## Funcionalidades
- **Crea tu perfil**: el usuario elige y califica películas (entrada de usuario) y recibe recomendaciones explicadas.
- **Usuario existente**: historial, géneros favoritos, top-N por modelo, aporte de cada modelo al híbrido y comparación de los 5 modelos.
- **Películas similares**: vecinos por CF, contenido y espacio latente SVD.
- **Evaluación**: Precision@K, Recall@K, curva de entrenamiento de NeuMF y búsqueda de pesos del híbrido.
