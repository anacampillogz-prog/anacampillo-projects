# Missing Data Imputation in Psychometric Assessment

Repositorio asociado al Trabajo Fin de Máster:

**“Evaluación Comparativa de Métodos de Imputación en Datos Psicométricos de RRHH mediante Simulación Monte Carlo”**

## Descripción

Este proyecto analiza el rendimiento de distintos métodos de imputación de valores perdidos en datos psicométricos aplicados a evaluación de Recursos Humanos.  

El estudio utiliza simulaciones Monte Carlo para comparar métodos clásicos y técnicas basadas en deep learning bajo diferentes mecanismos de missing (MCAR, MAR y MNAR), porcentajes de datos ausentes y niveles de complejidad factorial.

## Objetivos

- Simular instrumentos psicométricos con distintas estructuras factoriales.
- Generar mecanismos de missing MCAR, MAR y MNAR.
- Comparar distintos métodos de imputación.
- Evaluar precisión, sesgo y preservación estructural.
- Analizar implicaciones aplicadas en evaluación psicológica y RRHH.

## Métodos de imputación evaluados

- Mean Imputation
- MICE (Multiple Imputation by Chained Equations)
- Denoising Autoencoder (DAE)

## Métricas utilizadas

- RMSE
- MAE
- Mean Error (ME)
- Preservación correlacional
- Análisis estructural mediante UMAP

## Tecnologías

- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn

## Estructura del repositorio

```text
TFM_missing_data.ipynb
figures/
results/
