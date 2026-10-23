# Predicción de Precio de Vehículos con Regresión

Flujo completo de Machine Learning para estimar el precio de venta de un vehículo a partir de sus características, comparando **Regresión Lineal** y **Regresión Polinómica** (grados 1 a 4) y analizando el sobreajuste.

Actividad 2 de **Algoritmos de Aprendizaje Automático** · Universidad Tecmilenio.

📓 **Notebook (Google Colab):** [Abrir en Colab](TU_LINK_DE_COLAB)

## Datos

- [`data/vehiculos_actividad2.csv`](data/vehiculos_actividad2.csv): 1,000 vehículos × 16 variables, sin valores nulos.
- Numéricas: año, kilometraje, potencia (hp), cilindrada, puertas, pasajeros, consumo, batería (kWh), autonomía, puntajes de seguridad y equipamiento.
- Categóricas: marca, transmisión, tipo de combustible y tipo de vehículo.
- Variable objetivo: `precio`.
- En el análisis exploratorio, `potencia_hp` es la variable más correlacionada con el precio (positiva) y `kilometraje` la más negativa.

## Estructura

```
.
├── prediccion_precio_vehiculos.ipynb   # Notebook con todo el análisis
├── data/
│   └── vehiculos_actividad2.csv        # Dataset (1,000 vehículos)
└── README.md
```

## Metodología

1. Análisis exploratorio (distribución del precio, correlaciones).
2. Preprocesamiento: One-Hot Encoding (`drop_first=True`) y `StandardScaler`, ajustados solo con entrenamiento (sin data leakage).
3. División 80/20 con `random_state=42`.
4. Baseline con `DummyRegressor` (promedio).
5. Regresión lineal y polinómica (grados 1–4, solo sobre variables numéricas).
6. Validación cruzada K-Fold (k=5).
7. Métricas: MAE, RMSE y R², más análisis de residuales.

## Resultados (conjunto de prueba)

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Baseline | 255,598 | 328,199 | -0.0057 |
| **Regresión lineal** | 29,274 | 37,327 | 0.987 |
| Polinómica grado 2 | 30,541 | 37,544 | 0.9868 |
| Polinómica grado 3 | 49,960 | 66,368 | 0.9589 |
| Polinómica grado 4 | 550,476 | 1,565,597 | -21.88 |

- La **regresión lineal** fue el mejor equilibrio entre precisión y generalización: R² de train y test casi idénticos y R² promedio de 0.9854 en validación cruzada.
- Del grado 3 en adelante aparece sobreajuste claro; en grado 4 el R² de entrenamiento llega a 1.0 mientras el de prueba se desploma.
- Los residuales se distribuyen de forma pareja alrededor de cero, sin patrón de embudo.

## Ejemplo de predicción

Toyota SUV 2025, automático, gasolina, 210 hp, 5,000 km, seguridad 8.5/10 y equipamiento 8.0/10 → precio estimado de **$1,181,663 MXN**.

## Cómo reproducirlo

1. Abre el notebook en Google Colab.
2. Sube `data/vehiculos_actividad2.csv` (incluido en este repositorio) al entorno de Colab.
3. Ejecuta las celdas en orden.

Dependencias: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.

## Autor

**Axcel Francisco Salazar Alejo** · [@Axcelini](https://github.com/Axcelini)

Docente: Frederik Timoteo Castillo Pedraza
