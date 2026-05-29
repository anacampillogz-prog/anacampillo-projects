# Imputación de Valores Perdidos en Evaluación Psicométrica

Repositorio asociado al Trabajo Fin de Máster:

**“Protocolo de Imputación de Valores Perdidos en Evaluación Psicométrica de Competencias: Propuesta de Mejora para Entornos Aplicados de Selección de Personal”**

## Descripción

Este repositorio contiene el código desarrollado para el Trabajo Fin de Máster realizado en el marco de unas prácticas externas en el ámbito de la psicometría aplicada a la evaluación de competencias.

El proyecto surge a partir de la identificación de limitaciones en los procedimientos tradicionales de imputación de valores perdidos utilizados en contextos de evaluación psicométrica. Como propuesta de mejora, se evalúan diferentes métodos de imputación mediante un estudio de simulación diseñado para reproducir las características habituales de los instrumentos empleados en selección de personal y gestión del talento.

## Objetivos

* Evaluar el rendimiento de distintos métodos de imputación de valores perdidos.
* Analizar su comportamiento bajo diferentes mecanismos de ausencia de datos (MCAR, MAR y MNAR).
* Estudiar el impacto de la imputación sobre la precisión de las puntuaciones recuperadas.
* Evaluar la preservación de la estructura psicométrica de los instrumentos.
* Analizar las implicaciones prácticas de cada método en procesos de evaluación de competencias.
* Fundamentar una propuesta de mejora aplicable a entornos profesionales de evaluación psicométrica.

## Métodos evaluados

### Método de referencia

* Imputación por la media del ítem

### Imputación iterativa

* IterativeImputer (inspirado conceptualmente en MICE)

### Aprendizaje profundo

* Denoising Autoencoder (DAE)

## Diseño de simulación

La validación se realizó mediante simulación Monte Carlo utilizando datos psicométricos sintéticos con características similares a las observadas en instrumentos de evaluación de competencias:

* Escenarios factoriales simples y complejos.
* Escalas tipo Likert de 1 a 5 puntos.
* Mecanismos de ausencia MCAR, MAR y MNAR.
* Tasas de valores perdidos del 10% y 30%.
* 30 repeticiones independientes por condición experimental.

## Métricas de evaluación

* RMSE
* MAE
* NRMSE
* Mean Error (ME)
* Preservación de correlaciones
* Análisis de reclasificación competencial
* Visualización estructural mediante UMAP

## Tecnologías utilizadas

* Python
* NumPy
* Pandas
* Scikit-learn
* TensorFlow / Keras
* UMAP-learn
* Matplotlib
* Seaborn

## Contenido del repositorio

* Simulación de datos psicométricos.
* Generación de mecanismos MCAR, MAR y MNAR.
* Implementación de los métodos de imputación.
* Evaluación mediante métricas de error y preservación estructural.
* Generación de tablas y figuras utilizadas en la memoria del TFM.

## Resultados generales

Los resultados obtenidos sugieren que los métodos de imputación iterativa ofrecen ventajas respecto a la imputación por la media en los escenarios más habituales de datos perdidos, mejorando la precisión de las imputaciones y reduciendo sesgos sistemáticos.

