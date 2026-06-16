"""
TFM: Imputación de valores perdidos en datos psicométricos
Comparación: Baseline (media/moda) vs MICE vs Denoising Autoencoder
Incluye: MCAR, MAR, MNAR | UMAP | Curva de pérdida | Tablas y figuras para TFM
Figuras: F1 RMSE interacción | F2 ME sesgo | F3 NRMSE boxplot |
         F4 MAD correlaciones | F5 Loss curves | F6 UMAP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import umap

# CONFIGURACIÓN GLOBAL

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

N_PARTICIPANTS  = 750
N_REPETITIONS   = 30
N_EPOCHS        = 20
MISSING_RATES   = [0.10, 0.30]
MECHANISMS      = ['MCAR', 'MAR', 'MNAR']
LIKERT_MIN      = 1
LIKERT_MAX      = 5
LIKERT_RANGE    = LIKERT_MAX - LIKERT_MIN  # 4, para normalizar RMSE

# Escenarios de dimensionalidad
SCENARIOS = {
    'simple':  {'n_factors': 6,  'items_per_factor': 5},
    'complex': {'n_factors': 12, 'items_per_factor': 5},
}

# Colores para figuras
COLORS = {
    'baseline':    '#a8dadc',
    'mice':        '#1d3557',
    'autoencoder': '#e63946',
    'original':    '#2d6a4f',
    'missing':     '#f4a261',
}

# 1. SIMULACIÓN DE DATOS PSICOMÉTRICOS

def simulate_psychometric_data(n_participants, n_factors, items_per_factor, seed=None):
    """
    Simula datos psicométricos con estructura factorial realista.
    - Cargas factoriales primarias: 0.50–0.80
    - Cargas cruzadas: el primer ítem de cada factor tiene una carga secundaria
      de 0.20–0.25 en el factor adyacente (20% de los ítems; Marsh et al., 2014).
    - Correlaciones entre factores: 0.20–0.35
    - Varianza residual ampliada mediante un factor de inflación de 1,4 sobre
      la varianza residual derivada de las comunalidades (1 − h²), lo que
      representa un incremento del 40% respecto al residuo natural del modelo
      factorial. Este factor compensa el aumento de comunalidad introducido por
      las cargas cruzadas y añade varianza única atribuible a factores menores
      no modelados y error de medida no explicado por la estructura factorial,
      siguiendo criterios conservadores habituales en simulación psicométrica
      (Pereira et al., 2020; Gondara y Wang, 2018).
    - Discretización Likert 1–5 con distribución asimétrica negativa que refleja
      los sesgos de aquiescencia y deseabilidad social documentados en evaluación
      de competencias en selección de personal:
        1 ≈ 10%  (<p10 de la normal estándar, corte −1.28)
        2 ≈ 15%  (p10–p25, corte −0.67)
        3 ≈ 25%  (p25–p50, corte  0.00)
        4 ≈ 30%  (p50–p80, corte +0.84)
        5 ≈ 20%  (>p80)
    """
    if seed is not None:
        np.random.seed(seed)

    n_items = n_factors * items_per_factor

    # Matriz de correlaciones entre factores
    factor_corr = np.full((n_factors, n_factors), 0.275)
    np.fill_diagonal(factor_corr, 1.0)
    # Añadir variabilidad realista
    noise = np.random.uniform(-0.075, 0.075, (n_factors, n_factors))
    noise = (noise + noise.T) / 2
    np.fill_diagonal(noise, 0)
    factor_corr = np.clip(factor_corr + noise, 0.20, 0.35)
    np.fill_diagonal(factor_corr, 1.0)

    # Factores latentes correlacionados
    L = np.linalg.cholesky(factor_corr)
    factors_raw = np.random.randn(n_participants, n_factors)
    factors = factors_raw @ L.T

    # Cargas factoriales primarias (0.50–0.80)
    loadings = np.random.uniform(0.50, 0.80, (n_items, n_factors))

    # CAMBIO 1 — Cargas cruzadas (Marsh et al., 2014)
    # El primer ítem de cada factor (i = 0, 5, 10, …) recibe una carga secundaria
    # de 0.20–0.25 en el factor adyacente (módulo n_factors).
    # Esto representa el 20% de los ítems con complejidad factorial secundaria
    # no trivial, conservador respecto al 30–40% habitual en la literatura.
    loading_matrix = np.zeros((n_items, n_factors))
    for i in range(n_items):
        factor_idx = i // items_per_factor
        loading_matrix[i, factor_idx] = loadings[i, factor_idx]
        # Primer ítem de cada factor → carga cruzada en factor adyacente
        if i % items_per_factor == 0:
            adjacent_factor = (factor_idx + 1) % n_factors
            cross_loading = np.random.uniform(0.20, 0.25)
            loading_matrix[i, adjacent_factor] = cross_loading

    # CAMBIO 3 — Varianza residual ampliada (×1.4 = +40% sobre el residuo base)
    # El factor de inflación 1.4 actúa sobre la varianza residual natural (1 − h²):
    #   - Compensa el aumento de comunalidad producido por las cargas cruzadas.
    #   - Añade varianza única atribuible a factores menores no modelados
    #     y error de medida no capturado por la estructura factorial.
    # Rango resultante de varianza residual: ~0.43–1.05 según la carga primaria.
    error_var = 1 - np.sum(loading_matrix**2, axis=1)
    error_var = np.clip(error_var, 0.1, None) * 1.4   # factor de inflación de ruido
    errors = np.random.randn(n_participants, n_items) * np.sqrt(error_var)
    data_continuous = factors @ loading_matrix.T + errors

    # CAMBIO 2 — Discretización Likert con distribución asimétrica negativa
    # Cortes fijos sobre la variable estandarizada que reproducen la distribución
    # típica de competencias laborales en contextos de selección (aquiescencia y
    # deseabilidad social): 1≈10%, 2≈15%, 3≈25%, 4≈30%, 5≈20%.
    LIKERT_CUTS = np.array([-1.28, -0.67, 0.00, 0.84])  # cuantiles N(0,1)

    data_likert = np.zeros_like(data_continuous, dtype=int)
    for j in range(n_items):
        col = data_continuous[:, j]
        # Estandarizar columna para aplicar los cortes sobre N(0,1)
        col_z = (col - col.mean()) / (col.std() + 1e-8)
        data_likert[:, j] = np.digitize(col_z, LIKERT_CUTS) + 1

    data_likert = np.clip(data_likert, LIKERT_MIN, LIKERT_MAX)
    columns = [f'F{(i//items_per_factor)+1}_I{(i%items_per_factor)+1}' for i in range(n_items)]
    return pd.DataFrame(data_likert, columns=columns), loading_matrix

# 2. INTRODUCCIÓN DE VALORES PERDIDOS

def introduce_missing(data, mechanism, missing_rate, seed=None):
    """
    Introduce valores perdidos según el mecanismo especificado.

    MCAR: completamente aleatorio
    MAR:  probabilidad de missing depende de otros ítems observados
    MNAR: probabilidad de missing depende del propio valor (valores bajos → más missing)
    """
    if seed is not None:
        np.random.seed(seed)

    data_missing = data.copy().astype(float)
    n, p = data.shape

    if mechanism == 'MCAR':
        mask = np.random.rand(n, p) < missing_rate
        data_missing[mask] = np.nan

    elif mechanism == 'MAR':
        n_anchor = max(1, p // 6)
        anchor_cols = list(range(n_anchor))
        anchor_score = data.iloc[:, anchor_cols].mean(axis=1).values
        # Estandarizar para centrar la logistica en la media muestral
        anchor_score = (anchor_score - anchor_score.mean()) / (anchor_score.std() + 1e-8)
        # Funcion logistica con pendiente k=2.5: diferencia clara entre sujetos
        # con puntuacion alta vs baja en el factor ancla (~3.5x de ratio de probabilidades)
        base_prob = 1 / (1 + np.exp(-2.5 * anchor_score))
        for j in range(p):
            if j in anchor_cols:
                # Los items ancla no condicionan su propio missing: usar resto del primer factor
                alt_cols = [c for c in anchor_cols if c != j] or list(range(n_anchor, min(n_anchor*2, p)))
                alt_score = data.iloc[:, alt_cols].mean(axis=1).values
                alt_score = (alt_score - alt_score.mean()) / (alt_score.std() + 1e-8)
                prob = 1 / (1 + np.exp(-2.5 * alt_score))
            else:
                prob = base_prob.copy()
            # Reescalar para alcanzar el missing_rate global deseado
            prob = prob / prob.mean() * missing_rate
            prob = np.clip(prob, 0, 1)
            mask = np.random.rand(n) < prob
            data_missing.iloc[mask, j] = np.nan

    elif mechanism == 'MNAR':
        # Valores bajos (≤ 2) tienen mayor probabilidad de ser missing
        # Simula que personas con puntuaciones bajas en un rasgo tienden a no responder
        for j in range(p):
            col = data.iloc[:, j].values
            prob = np.where(col <= 2, missing_rate * 1.8, missing_rate * 0.4)
            prob = np.clip(prob, 0, 1)
            # Ajustar para que el missing_rate global sea aproximadamente el deseado
            scale = missing_rate / prob.mean()
            prob = np.clip(prob * scale, 0, 1)
            mask = np.random.rand(n) < prob
            data_missing.iloc[mask, j] = np.nan

    return data_missing

# 3. MÉTODOS DE IMPUTACIÓN

# --- 3.1 Baseline: Media/Moda ---

def impute_baseline(data_missing):
    """Imputa con la media (aproximación para Likert en baseline)."""
    imputer = SimpleImputer(strategy='mean')
    imputed = imputer.fit_transform(data_missing)
    # Redondear y clipear a rango Likert
    imputed = np.clip(np.round(imputed), LIKERT_MIN, LIKERT_MAX)
    return pd.DataFrame(imputed, columns=data_missing.columns)


# --- 3.2 Iterative Imputer ---

def impute_mice(data_missing, seed=None):
    """Imputación múltiple por ecuaciones encadenadas (MICE)."""
    imputer = IterativeImputer(
        max_iter=10,
        random_state=seed if seed is not None else SEED,
        min_value=LIKERT_MIN,
        max_value=LIKERT_MAX
    )
    imputed = imputer.fit_transform(data_missing)
    imputed = np.clip(np.round(imputed), LIKERT_MIN, LIKERT_MAX)
    return pd.DataFrame(imputed, columns=data_missing.columns)


# --- 3.3 Denoising Autoencoder ---

def build_autoencoder(n_items, encoding_dim=None):
    """
    Construye un Denoising Autoencoder fully connected.
    La capa bottleneck realiza reducción de dimensionalidad implícita.
    encoding_dim: dimensión del espacio latente (por defecto: n_items // 3)
    """
    if encoding_dim is None:
        encoding_dim = max(n_items // 3, 4)

    inputs = keras.Input(shape=(n_items,))

    # Encoder — reducción de dimensionalidad
    x = layers.Dense(n_items, activation='relu')(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(n_items // 2, activation='relu')(x)
    encoded = layers.Dense(encoding_dim, activation='relu', name='bottleneck')(x)

    # Decoder — reconstrucción
    x = layers.Dense(n_items // 2, activation='relu')(encoded)
    x = layers.Dense(n_items, activation='relu')(x)
    # Salida en rango [0,1] (los datos se normalizan antes)
    outputs = layers.Dense(n_items, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model


def impute_autoencoder(data_missing, n_epochs=N_EPOCHS, seed=None):
    """
    Imputa usando un Denoising Autoencoder.
    Devuelve los datos imputados y el historial de pérdida por época.
    """
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    n, p = data_missing.shape

    # Normalizar a [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Fit solo con valores observados (usando la media para los NaN temporalmente)
    temp_fill = data_missing.fillna(data_missing.mean())
    scaler.fit(temp_fill)

    data_scaled = scaler.transform(temp_fill)
    missing_mask = data_missing.isna().values

    # Inicializar NaN con la media normalizada
    data_input = data_scaled.copy()

    # Construir modelo
    model = build_autoencoder(p)

    # Entrenamiento iterativo: el modelo aprende a reconstruir a pesar del ruido
    history_loss = []
    for epoch in range(n_epochs):
        # Añadir ruido gaussiano a la entrada (denoising)
        noise = np.random.normal(0, 0.1, data_input.shape)
        data_noisy = np.clip(data_input + noise, 0, 1)

        hist = model.fit(
            data_noisy, data_input,
            epochs=1,
            batch_size=32,
            verbose=0
        )
        history_loss.append(hist.history['loss'][0])

        # Actualizar los valores imputados tras cada época
        reconstructed = model.predict(data_input, verbose=0)
        data_input[missing_mask] = reconstructed[missing_mask]

    # Desnormalizar
    data_reconstructed = scaler.inverse_transform(data_input)
    data_reconstructed = np.clip(np.round(data_reconstructed), LIKERT_MIN, LIKERT_MAX)

    return pd.DataFrame(data_reconstructed, columns=data_missing.columns), history_loss


# 4. MÉTRICAS DE EVALUACIÓN

def compute_metrics(original, imputed, missing_mask):
    """
    Calcula RMSE, MAE, RMSE normalizado y sesgo medio (ME) solo sobre los valores
    que eran missing.

    El sesgo medio (ME = mean(imputado - original)) detecta sobreestimación o
    subestimación sistemática, complementando al RMSE que al ser simétrico no
    distingue el signo del error. Bajo MNAR con valores bajos ausentes, un ME
    positivo indica sobreestimación sistemática (Van Buuren, 2018;
    Madley-Dowd et al., 2019).
    """
    orig_vals    = original.values[missing_mask]
    imputed_vals = imputed.values[missing_mask]

    if len(orig_vals) == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'nrmse': np.nan, 'me': np.nan}

    rmse  = np.sqrt(np.mean((orig_vals - imputed_vals) ** 2))
    mae   = np.mean(np.abs(orig_vals - imputed_vals))
    nrmse = rmse / LIKERT_RANGE  # normalizado por rango de la escala
    me    = np.mean(imputed_vals - orig_vals)  # positivo = sobreestimación

    return {'rmse': rmse, 'mae': mae, 'nrmse': nrmse, 'me': me}


def correlation_matrix_mad(original, imputed):
    """
    Diferencia media absoluta entre matrices de correlación original e imputada.
    Mide si la imputación preserva la estructura psicométrica del instrumento.
    """
    corr_orig   = original.corr().values
    corr_imputed = imputed.corr().values
    mask = ~np.eye(corr_orig.shape[0], dtype=bool)  # excluir diagonal
    return np.mean(np.abs(corr_orig[mask] - corr_imputed[mask]))



def compute_ci95(values):
    """Intervalo de confianza al 95% para la media."""
    n    = len(values)
    mean = np.mean(values)
    se   = np.std(values, ddof=1) / np.sqrt(n)
    ci   = 1.96 * se
    return mean, ci


# 4b. TESTS DE WILCOXON PAREADOS

def run_wilcoxon_tests(df_results):
    """
    Pruebas de Wilcoxon pareadas sobre las 30 réplicas Monte Carlo para las
    comparaciones clave entre métodos.

    Las comparaciones elegidas responden a las hipótesis centrales del TFM:
      - MICE vs. Baseline bajo MCAR y MAR (donde MICE debería ser superior)
      - MICE vs. Baseline bajo MNAR 30% (donde el baseline obtiene menor RMSE)
      - MICE vs. DAE bajo MCAR y MAR (para confirmar la superioridad de MICE)

    Nota: se usa la corrección de Benjamini-Hochberg (FDR) para el conjunto
    de comparaciones múltiples (8 tests), por ser más potente que Bonferroni
    y adecuada para datos correlacionados de simulación.

    Returns
    -------
    df_wilcoxon : pd.DataFrame con columnas:
        escenario, mecanismo, missing_pct, comparacion, W, p_raw, p_adj,
        significativo, direccion, rmse_A_media, rmse_B_media
    """
    from scipy.stats import wilcoxon
    from itertools import product

    comparisons = [
        # (método_A, método_B, mecanismos, missing_pcts, descripción)
        ('mice', 'baseline', ['MCAR', 'MAR'],  [10, 30], 'MICE vs Baseline'),
        ('mice', 'baseline', ['MNAR'], [10, 30], 'MICE vs Baseline'),  # caso inverso
        ('mice', 'autoencoder', ['MCAR', 'MAR'], [10, 30], 'MICE vs DAE'),
    ]

    rows = []
    for scenario in ['simple', 'complex']:
        df_s = df_results[df_results['scenario'] == scenario]
        for method_a, method_b, mechs, missing_pcts, desc in comparisons:
            for mech, mpct in product(mechs, missing_pcts):
                vals_a = df_s[
                    (df_s['method'] == method_a) &
                    (df_s['mechanism'] == mech) &
                    (df_s['missing_pct'] == mpct)
                ]['rmse'].values

                vals_b = df_s[
                    (df_s['method'] == method_b) &
                    (df_s['mechanism'] == mech) &
                    (df_s['missing_pct'] == mpct)
                ]['rmse'].values

                if len(vals_a) < 2 or len(vals_b) < 2:
                    continue

                # Wilcoxon sobre diferencias pareadas (misma réplica)
                diffs = vals_a - vals_b
                if np.all(diffs == 0):
                    W, p = np.nan, 1.0
                else:
                    W, p = wilcoxon(diffs, alternative='two-sided')

                direction = (
                    f"{method_a.upper()} < {method_b.upper()}"
                    if np.mean(vals_a) < np.mean(vals_b)
                    else f"{method_a.upper()} > {method_b.upper()}"
                )

                rows.append({
                    'escenario':    scenario,
                    'mecanismo':    mech,
                    'missing_pct':  mpct,
                    'comparacion':  desc,
                    'metodo_A':     method_a,
                    'metodo_B':     method_b,
                    'W':            W,
                    'p_raw':        p,
                    'rmse_A_media': np.mean(vals_a),
                    'rmse_B_media': np.mean(vals_b),
                    'direccion':    direction,
                })

    df_w = pd.DataFrame(rows)

    # Corrección FDR (Benjamini-Hochberg) sobre todos los p-valores
    from scipy.stats import false_discovery_control
    try:
        df_w['p_adj'] = false_discovery_control(df_w['p_raw'].fillna(1).values)
    except AttributeError:
        # scipy < 1.11: implementación manual de BH
        p_vals = df_w['p_raw'].fillna(1).values
        n = len(p_vals)
        rank = np.argsort(p_vals)
        p_adj = np.empty(n)
        p_adj[rank] = p_vals[rank] * n / (np.arange(n) + 1)
        # Cumulative minimum from the right (monotone enforcement)
        p_adj = np.minimum.accumulate(p_adj[::-1])[::-1]
        df_w['p_adj'] = np.clip(p_adj, 0, 1)

    df_w['significativo'] = df_w['p_adj'] < 0.05

    return df_w


def print_wilcoxon_summary(df_w):
    """
    Imprime un resumen legible de los tests de Wilcoxon y genera frases
    listas para insertar en el texto del TFM.
    """
    print("\n" + "=" * 70)
    print("TESTS DE WILCOXON PAREADOS (30 réplicas MC) — corrección FDR (BH)")
    print("=" * 70)

    # Tabla completa
    cols_show = ['escenario', 'mecanismo', 'missing_pct', 'comparacion',
                 'direccion', 'W', 'p_raw', 'p_adj', 'significativo']
    print(df_w[cols_show].to_string(index=False))

    # Frases para el TFM
    print("\n" + "-" * 70)
    print("FRASES PARA INSERTAR EN EL TFM:")
    print("-" * 70)

    for _, row in df_w.iterrows():
        sig_str = "p_adj < 0,05" if row['significativo'] else f"p_adj = {row['p_adj']:.3f}, n.s."
        direction_label = "inferior" if "<" in row['direccion'] else "superior"
        print(
            f"[{row['escenario'].upper()} | {row['mecanismo']} {row['missing_pct']}% | "
            f"{row['comparacion']}] "
            f"RMSE medio: {row['metodo_A'].upper()}={row['rmse_A_media']:.3f} vs "
            f"{row['metodo_B'].upper()}={row['rmse_B_media']:.3f}. "
            f"Wilcoxon pareado: W={row['W']:.0f}, {sig_str}. "
            f"Dirección: {row['metodo_A'].upper()} {direction_label} al {row['metodo_B'].upper()}."
        )

    print("=" * 70)



# 5. SIMULACIÓN MONTE CARLO PRINCIPAL

# 5b. ANÁLISIS DE RECLASIFICACIÓN COMPETENCIAL

def compute_factor_scores(data, n_factors, items_per_factor):
    """
    Calcula la puntuación media de cada sujeto en cada factor
    sumando los ítems correspondientes.
    """
    scores = np.zeros((len(data), n_factors))
    for f in range(n_factors):
        cols = [f'F{f+1}_I{i+1}' for i in range(items_per_factor)]
        scores[:, f] = data[cols].mean(axis=1).values
    return scores


def assign_levels(scores, cutpoints):
    """
    Asigna niveles 1–5 a cada puntuación según los puntos de corte.
    cutpoints: array de 4 umbrales que delimitan los 5 niveles.
    """
    return np.digitize(scores, cutpoints) + 1


def reclassification_rate(levels_orig, levels_imputed):
    """
    Porcentaje de sujetos (promediado entre factores) cuyo nivel
    competencial cambia tras la imputación respecto al original.
    """
    # levels_orig y levels_imputed: (n_subjects, n_factors)
    changed = (levels_orig != levels_imputed)          # bool matrix
    # Porcentaje de sujetos que cambian en al menos un factor
    any_change = changed.any(axis=1).mean() * 100
    # Porcentaje medio de cambios por factor (tasa de cambio global)
    per_factor  = changed.mean(axis=0) * 100           # (n_factors,)
    mean_factor = per_factor.mean()
    return any_change, mean_factor


def run_reclassification_analysis(n_factors, items_per_factor, mechanisms_target,
                                  missing_rate=0.30, n_reps=30):
    """
    Para las condiciones MAR y MNAR con 30% de missing, calcula el porcentaje
    de sujetos cuya clasificación competencial (niveles 1–5) cambia según
    el método de imputación respecto a los datos originales.

    Puntos de corte: quintiles de la distribución original (20, 40, 60, 80 percentil),
    calculados por repetición para reflejar la distribución muestral real.
    """
    results = []
    scenario_name = 'simple'   # escenario de referencia (6F, más impactante por ítem)
    n_items = n_factors * items_per_factor
    condition_idx_offset = 999  # offset para evitar colisión de seeds con simulación principal

    for mechanism in mechanisms_target:
        print(f"\n[Reclasificación] Mecanismo: {mechanism} | Missing: {int(missing_rate*100)}%")
        for rep in range(n_reps):
            rep_seed = SEED + rep * 1000 + condition_idx_offset

            # Datos originales
            data_orig, _ = simulate_psychometric_data(
                N_PARTICIPANTS, n_factors, items_per_factor, seed=rep_seed
            )
            # Introducir missing
            data_miss = introduce_missing(data_orig, mechanism, missing_rate, seed=rep_seed + 1)

            # Imputaciones
            imp_bl   = impute_baseline(data_miss)
            imp_mice = impute_mice(data_miss, seed=rep_seed + 2)
            imp_ae, _ = impute_autoencoder(data_miss, n_epochs=N_EPOCHS, seed=rep_seed + 3)

            # Puntuaciones factoriales
            scores_orig = compute_factor_scores(data_orig, n_factors, items_per_factor)
            scores_bl   = compute_factor_scores(imp_bl,   n_factors, items_per_factor)
            scores_mice = compute_factor_scores(imp_mice, n_factors, items_per_factor)
            scores_ae   = compute_factor_scores(imp_ae,   n_factors, items_per_factor)

            # Puntos de corte basados en la distribución original (quintiles por factor)
            # Se calcula por factor y se aplica la misma escala a todos los métodos
            levels_orig = np.zeros_like(scores_orig, dtype=int)
            levels_bl   = np.zeros_like(scores_bl,   dtype=int)
            levels_mice = np.zeros_like(scores_mice, dtype=int)
            levels_ae   = np.zeros_like(scores_ae,   dtype=int)

            for f in range(n_factors):
                cuts = np.percentile(scores_orig[:, f], [20, 40, 60, 80])
                levels_orig[:, f] = assign_levels(scores_orig[:, f], cuts)
                levels_bl[:, f]   = assign_levels(scores_bl[:, f],   cuts)
                levels_mice[:, f] = assign_levels(scores_mice[:, f], cuts)
                levels_ae[:, f]   = assign_levels(scores_ae[:, f],   cuts)

            # Reclasificación respecto al original
            for method, levels_imp in [
                ('Baseline',    levels_bl),
                ('MICE',        levels_mice),
                ('Autoencoder', levels_ae),
            ]:
                any_chg, mean_chg = reclassification_rate(levels_orig, levels_imp)
                results.append({
                    'mechanism':       mechanism,
                    'missing_pct':     int(missing_rate * 100),
                    'repetition':      rep,
                    'method':          method,
                    'pct_any_change':  any_chg,    # % sujetos con cambio en ≥1 factor
                    'pct_mean_factor': mean_chg,   # % medio de cambio por factor
                })

            if rep % 10 == 0:
                print(f"  Rep {rep+1}/{n_reps} completada")

    return pd.DataFrame(results)


def generate_reclassification_table(df_reclasif):
    """
    Genera tabla resumen con media e IC95% del % de reclasificación.
    """
    rows = []
    for mechanism in df_reclasif['mechanism'].unique():
        for method in ['Baseline', 'MICE', 'Autoencoder']:
            sub = df_reclasif[
                (df_reclasif['mechanism'] == mechanism) &
                (df_reclasif['method'] == method)
            ]
            m_any,  ci_any  = compute_ci95(sub['pct_any_change'].values)
            m_fact, ci_fact = compute_ci95(sub['pct_mean_factor'].values)
            rows.append({
                'Mecanismo':                  mechanism,
                'Método':                     method,
                '% sujetos reclasificados (≥1 factor)': f"{m_any:.1f} ± {ci_any:.1f}%",
                '% cambio medio por factor':  f"{m_fact:.1f} ± {ci_fact:.1f}%",
            })
    df_table = pd.DataFrame(rows)
    df_table.to_csv('tabla_reclasificacion_competencial.csv', index=False)
    print("\nTabla de reclasificación guardada: tabla_reclasificacion_competencial.csv")
    print(df_table.to_string(index=False))
    return df_table


def run_simulation():
    """
    Ejecuta la simulación Monte Carlo completa.
    Devuelve un DataFrame con todos los resultados.
    """
    results = []
    all_loss_curves = {}  # Para curvas de pérdida del autoencoder

    scenario_names = list(SCENARIOS.keys())
    total_conditions = len(scenario_names) * len(MECHANISMS) * len(MISSING_RATES)
    condition_idx = 0

    for scenario_name, scenario_params in SCENARIOS.items():
        n_factors       = scenario_params['n_factors']
        items_per_factor = scenario_params['items_per_factor']
        n_items         = n_factors * items_per_factor

        for mechanism in MECHANISMS:
            for missing_rate in MISSING_RATES:
                condition_idx += 1
                print(f"\n[{condition_idx}/{total_conditions}] "
                      f"Escenario: {scenario_name} ({n_factors}F×{items_per_factor}I) | "
                      f"Mecanismo: {mechanism} | Missing: {missing_rate*100:.0f}%")

                condition_key = f"{scenario_name}_{mechanism}_{int(missing_rate*100)}"
                loss_curves_condition = []

                for rep in range(N_REPETITIONS):
                    rep_seed = SEED + rep * 1000 + condition_idx * 100

                    # Simular datos originales
                    data_orig, _ = simulate_psychometric_data(
                        N_PARTICIPANTS, n_factors, items_per_factor, seed=rep_seed
                    )

                    # Introducir missing
                    data_miss = introduce_missing(
                        data_orig, mechanism, missing_rate, seed=rep_seed + 1
                    )
                    missing_mask = data_miss.isna().values

                    # ── Imputación Baseline ──
                    imp_baseline = impute_baseline(data_miss)
                    m_bl = compute_metrics(data_orig, imp_baseline, missing_mask)
                    corr_bl = correlation_matrix_mad(data_orig, imp_baseline)

                    # ── Imputación MICE ──
                    imp_mice = impute_mice(data_miss, seed=rep_seed + 2)
                    m_mice = compute_metrics(data_orig, imp_mice, missing_mask)
                    corr_mice = correlation_matrix_mad(data_orig, imp_mice)

                    # ── Imputación Autoencoder ──
                    imp_ae, loss_curve = impute_autoencoder(
                        data_miss, n_epochs=N_EPOCHS, seed=rep_seed + 3
                    )
                    m_ae = compute_metrics(data_orig, imp_ae, missing_mask)
                    corr_ae = correlation_matrix_mad(data_orig, imp_ae)
                    loss_curves_condition.append(loss_curve)

                    # Guardar resultados de esta repetición
                    for method, metrics, corr_mad in [
                        ('baseline',    m_bl,   corr_bl),
                        ('mice',        m_mice, corr_mice),
                        ('autoencoder', m_ae,   corr_ae),
                    ]:
                        results.append({
                            'scenario':    scenario_name,
                            'n_factors':   n_factors,
                            'n_items':     n_items,
                            'mechanism':   mechanism,
                            'missing_pct': int(missing_rate * 100),
                            'repetition':  rep,
                            'method':      method,
                            'rmse':        metrics['rmse'],
                            'mae':         metrics['mae'],
                            'nrmse':       metrics['nrmse'],
                            'me':          metrics['me'],
                            'corr_mad':    corr_mad,
                        })

                    if rep % 10 == 0:
                        print(f"  Rep {rep+1}/{N_REPETITIONS} completada")

                all_loss_curves[condition_key] = loss_curves_condition

    df_results = pd.DataFrame(results)
    return df_results, all_loss_curves


# 6. GENERACIÓN DE TABLAS PARA EL TFM

def generate_summary_table(df_results):
    """
    Genera tabla resumen con media e IC95% para cada condición y método.
    """
    summary_rows = []

    for method in ['baseline', 'mice', 'autoencoder']:
        df_m = df_results[df_results['method'] == method]
        groups = df_m.groupby(['scenario', 'n_factors', 'mechanism', 'missing_pct'])

        for (scenario, n_factors, mechanism, missing_pct), group in groups:
            rmse_mean,  rmse_ci  = compute_ci95(group['rmse'].values)
            mae_mean,   mae_ci   = compute_ci95(group['mae'].values)
            nrmse_mean, nrmse_ci = compute_ci95(group['nrmse'].values)
            me_mean,    me_ci    = compute_ci95(group['me'].values)
            corr_mean,  corr_ci  = compute_ci95(group['corr_mad'].values)

            summary_rows.append({
                'Escenario':   scenario,
                'Factores':    n_factors,
                'Mecanismo':   mechanism,
                'Missing (%)': missing_pct,
                'Método':      method.capitalize(),
                'RMSE':        f"{rmse_mean:.3f} ± {rmse_ci:.3f}",
                'MAE':         f"{mae_mean:.3f} ± {mae_ci:.3f}",
                'NRMSE':       f"{nrmse_mean:.3f} ± {nrmse_ci:.3f}",
                'ME':          f"{me_mean:.3f} ± {me_ci:.3f}",
                'Corr MAD':    f"{corr_mean:.3f} ± {corr_ci:.3f}",
            })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv('tabla_resumen_resultados.csv', index=False)
    print("\nTabla resumen guardada: tabla_resumen_resultados.csv")
    return df_summary


# 7. GENERACIÓN DE FIGURAS PARA EL TFM

def plot_rmse_interaction(df_results, save_path='figura1_rmse_interaccion.png'):
    """
    Figura de interacción: RMSE por método, mecanismo y % missing.
    Una subfigura por escenario de dimensionalidad.
    """
    methods  = ['baseline', 'mice', 'autoencoder']
    labels   = ['Baseline (Media)', 'MICE', 'Autoencoder']
    scenarios = df_results['scenario'].unique()

    fig, axes = plt.subplots(1, len(scenarios), figsize=(14, 5), sharey=True)
    fig.suptitle('RMSE por método, mecanismo y porcentaje de valores perdidos',
                 fontsize=13, fontweight='bold', y=1.02)

    for ax, scenario in zip(axes, scenarios):
        df_s = df_results[df_results['scenario'] == scenario]
        n_items = df_s['n_items'].iloc[0]
        n_factors = df_s['n_factors'].iloc[0]

        x_positions = np.arange(len(MECHANISMS))
        width = 0.13
        offsets = np.linspace(-0.28, 0.28, len(methods) * len(MISSING_RATES))
        idx = 0

        for method, label in zip(methods, labels):
            for missing_pct in [10, 30]:
                df_sub = df_s[(df_s['method'] == method) & (df_s['missing_pct'] == missing_pct)]
                means = []
                cis   = []
                for mech in MECHANISMS:
                    vals = df_sub[df_sub['mechanism'] == mech]['rmse'].values
                    m, ci = compute_ci95(vals)
                    means.append(m)
                    cis.append(ci)

                linestyle = '-' if missing_pct == 10 else '--'
                ax.errorbar(
                    x_positions + offsets[idx], means, yerr=cis,
                    label=f"{label} {missing_pct}%",
                    color=COLORS[method],
                    alpha=0.85 if missing_pct == 10 else 0.55,
                    linestyle=linestyle,
                    marker='o', markersize=5, capsize=3, linewidth=1.5
                )
                idx += 1

        ax.set_title(f'{scenario.capitalize()}\n({n_factors} factores, {n_items} ítems)',
                     fontsize=11)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(MECHANISMS)
        ax.set_xlabel('Mecanismo de valores perdidos')
        ax.set_ylabel('RMSE (media ± IC95%)')
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.12), fontsize=9, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figura guardada: {save_path}")


def plot_loss_curves(all_loss_curves, save_path='figura5_curva_loss_autoencoder.png'):
    """
    Curva de pérdida del autoencoder por condición.
    Muestra media ± banda de variabilidad entre repeticiones.
    """
    conditions = list(all_loss_curves.keys())
    n_cols = 4
    n_rows = int(np.ceil(len(conditions) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(14, n_rows * 3.2),
                             sharex=True)
    axes = axes.flatten()

    fig.suptitle('Curva de pérdida del Denoising Autoencoder por condición\n'
                 '(media ± desviación típica entre repeticiones)',
                 fontsize=12, fontweight='bold')

    for i, (condition, curves) in enumerate(all_loss_curves.items()):
        ax = axes[i]
        curves_arr = np.array(curves)  # shape: (n_reps, n_epochs)
        mean_curve  = curves_arr.mean(axis=0)
        std_curve   = curves_arr.std(axis=0)
        epochs      = np.arange(1, N_EPOCHS + 1)

        ax.plot(epochs, mean_curve, color=COLORS['autoencoder'], linewidth=2)
        ax.fill_between(epochs,
                        mean_curve - std_curve,
                        mean_curve + std_curve,
                        alpha=0.25, color=COLORS['autoencoder'])

        # Título limpio
        parts = condition.split('_')
        scenario, mech, missing = parts[0], parts[1], parts[2]
        ax.set_title(f"{scenario} | {mech} | {missing}% missing", fontsize=9)
        ax.set_xlabel('Época', fontsize=8)
        ax.set_ylabel('MSE Loss', fontsize=8)
        ax.grid(alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figura guardada: {save_path}")


def plot_umap_comparison(df_results, save_path='figura6_umap_comparacion.png'):
    """
    Proyección UMAP de datos originales, con missing, imputados por MICE
    y imputados por Autoencoder.
    Usa un escenario representativo (simple, MAR, 30% missing).
    """
    print("\nGenerando proyección UMAP (esto puede tardar unos segundos)...")

    # Usar seed fija para reproducibilidad
    np.random.seed(SEED)
    scenario_params = SCENARIOS['simple']
    n_factors        = scenario_params['n_factors']
    items_per_factor = scenario_params['items_per_factor']

    data_orig, _ = simulate_psychometric_data(
        N_PARTICIPANTS, n_factors, items_per_factor, seed=SEED
    )
    data_miss = introduce_missing(data_orig, 'MNAR', 0.30, seed=SEED + 1)

    # Imputaciones para UMAP
    imp_mice, _ = impute_mice(data_miss, seed=SEED + 2), None
    imp_mice = impute_mice(data_miss, seed=SEED + 2)
    imp_ae, _   = impute_autoencoder(data_miss, n_epochs=N_EPOCHS, seed=SEED + 3)

    # Rellenar missing con media para visualización del dataset incompleto
    data_miss_filled = data_miss.fillna(data_miss.mean())

    # Ajustar UMAP sobre datos originales
    reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=15, min_dist=0.1)
    emb_orig  = reducer.fit_transform(data_orig.values)
    emb_miss  = reducer.transform(data_miss_filled.values)
    emb_mice  = reducer.transform(imp_mice.values)
    emb_ae    = reducer.transform(imp_ae.values)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('Proyección UMAP: preservación de estructura psicométrica tras imputación\n'
                 '(Escenario: 6 factores, MNAR, 30% missing)',
                 fontsize=12, fontweight='bold')

    datasets = [
        (emb_orig, 'Datos originales',      COLORS['original']),
        (emb_miss, 'Con valores perdidos',   COLORS['missing']),
        (emb_mice, 'Imputado — MICE',        COLORS['mice']),
        (emb_ae,   'Imputado — Autoencoder', COLORS['autoencoder']),
    ]

    for ax, (emb, title, color) in zip(axes, datasets):
        ax.scatter(emb[:, 0], emb[:, 1],
                   c=color, alpha=0.5, s=8, linewidths=0)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('UMAP 1', fontsize=8)
        ax.set_ylabel('UMAP 2', fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figura guardada: {save_path}")


def plot_nrmse_by_method(df_results, save_path='figura3_nrmse_metodos.png'):
    """
    Boxplot de NRMSE por método para cada escenario.
    Permite comparar visualmente la distribución de errores normalizados.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle('RMSE normalizado por método y escenario\n'
                 '(valores menores = mejor imputación; escala: proporción del rango 1–5)',
                 fontsize=12, fontweight='bold')

    method_order  = ['baseline', 'mice', 'autoencoder']
    method_labels = ['Baseline\n(Media)', 'MICE', 'Autoencoder']
    palette       = [COLORS[m] for m in method_order]

    for ax, scenario in zip(axes, ['simple', 'complex']):
        df_s = df_results[df_results['scenario'] == scenario]
        n_items   = df_s['n_items'].iloc[0]
        n_factors = df_s['n_factors'].iloc[0]

        data_plot = [df_s[df_s['method'] == m]['nrmse'].values for m in method_order]
        bp = ax.boxplot(data_plot, patch_artist=True, widths=0.5,
                        medianprops=dict(color='black', linewidth=2))

        for patch, color in zip(bp['boxes'], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_title(f'{scenario.capitalize()}\n({n_factors} factores, {n_items} ítems)',
                     fontsize=11)
        ax.set_xticklabels(method_labels, fontsize=9)
        ax.set_ylabel('NRMSE (proporción del rango)', fontsize=9)
        ax.axhline(y=0.25, color='gray', linestyle=':', alpha=0.5,
                   label='25% del rango')
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figura guardada: {save_path}")


def plot_corr_preservation(df_results, save_path='figura4_preservacion_correlaciones.png'):
    """
    MAD de correlaciones por método y mecanismo.
    Evalúa si la imputación preserva la estructura psicométrica del instrumento.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle('Preservación de la estructura de correlaciones tras la imputación\n'
                 '(MAD menor = mejor preservación de la estructura psicométrica)',
                 fontsize=12, fontweight='bold')

    methods  = ['baseline', 'mice', 'autoencoder']
    labels   = ['Baseline', 'MICE', 'Autoencoder']
    x_pos    = np.arange(len(MECHANISMS))
    width    = 0.22

    for ax, scenario in zip(axes, ['simple', 'complex']):
        df_s      = df_results[df_results['scenario'] == scenario]
        n_items   = df_s['n_items'].iloc[0]
        n_factors = df_s['n_factors'].iloc[0]

        for j, (method, label) in enumerate(zip(methods, labels)):
            df_m  = df_s[df_s['method'] == method]
            means = []
            cis   = []
            for mech in MECHANISMS:
                vals = df_m[df_m['mechanism'] == mech]['corr_mad'].values
                m, ci = compute_ci95(vals)
                means.append(m)
                cis.append(ci)

            offset = (j - 1) * width
            ax.bar(x_pos + offset, means, width=width,
                   color=COLORS[method], alpha=0.8, label=label)
            ax.errorbar(x_pos + offset, means, yerr=cis,
                        fmt='none', color='black', capsize=3, linewidth=1.2)

        ax.set_title(f'{scenario.capitalize()}\n({n_factors} factores, {n_items} ítems)',
                     fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(MECHANISMS)
        ax.set_xlabel('Mecanismo de valores perdidos')
        ax.set_ylabel('MAD entre matrices de correlación (media ± IC95%)')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figura guardada: {save_path}")


def plot_me_bias(df_results, save_path='figura2_sesgo_medio_me.png'):
    """
    Sesgo medio (ME = imputado - original) por método, mecanismo y % missing.
    ME positivo indica sobreestimación sistemática; ME negativo, subestimación.
    Permite detectar el sesgo direccional que el RMSE no captura (Van Buuren, 2018).
    Solo se muestra el escenario simple para mayor claridad.
    """
    methods  = ['baseline', 'mice', 'autoencoder']
    labels   = ['Baseline (Media)', 'MICE', 'Autoencoder']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle('Sesgo medio de imputación (ME = imputado − original) por método y condición\n'
                 '(ME > 0 = sobreestimación; ME < 0 = subestimación; ME = 0 = sin sesgo)',
                 fontsize=12, fontweight='bold')

    for ax, scenario in zip(axes, ['simple', 'complex']):
        df_s = df_results[df_results['scenario'] == scenario]
        n_items   = df_s['n_items'].iloc[0]
        n_factors = df_s['n_factors'].iloc[0]

        x_positions = np.arange(len(MECHANISMS))
        width = 0.13
        offsets = np.linspace(-0.28, 0.28, len(methods) * len(MISSING_RATES))
        idx = 0

        for method, label in zip(methods, labels):
            for missing_pct in [10, 30]:
                df_sub = df_s[(df_s['method'] == method) & (df_s['missing_pct'] == missing_pct)]
                means = []
                cis   = []
                for mech in MECHANISMS:
                    vals = df_sub[df_sub['mechanism'] == mech]['me'].values
                    m, ci = compute_ci95(vals)
                    means.append(m)
                    cis.append(ci)

                linestyle = '-' if missing_pct == 10 else '--'
                ax.errorbar(
                    x_positions + offsets[idx], means, yerr=cis,
                    label=f"{label} {missing_pct}%",
                    color=COLORS[method],
                    alpha=0.85 if missing_pct == 10 else 0.55,
                    linestyle=linestyle,
                    marker='o', markersize=5, capsize=3, linewidth=1.5
                )
                idx += 1

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.set_title(f'{scenario.capitalize()}\n({n_factors} factores, {n_items} ítems)',
                     fontsize=11)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(MECHANISMS)
        ax.set_xlabel('Mecanismo de valores perdidos')
        ax.set_ylabel('ME medio (±IC95%)')
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.12), fontsize=9, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figura guardada: {save_path}")




# 8. EJECUCIÓN PRINCIPAL

if __name__ == '__main__':
    print("=" * 60)
    print("TFM: Imputación de valores perdidos en datos psicométricos")
    print("=" * 60)
    print(f"Configuración:")
    print(f"  Participantes:   {N_PARTICIPANTS}")
    print(f"  Repeticiones MC: {N_REPETITIONS}")
    print(f"  Épocas (AE):     {N_EPOCHS}")
    print(f"  Mecanismos:      {MECHANISMS}")
    print(f"  Missing rates:   {MISSING_RATES}")
    print("=" * 60)

    # ── Simulación Monte Carlo ──
    print("\n>>> Iniciando simulación Monte Carlo...\n")
    df_results, all_loss_curves = run_simulation()

    # Guardar resultados crudos
    df_results.to_csv('resultados_crudos.csv', index=False)
    print("\nResultados crudos guardados: resultados_crudos.csv")

    # ── Tablas ──
    print("\n>>> Generando tablas...")
    df_summary = generate_summary_table(df_results)
    print(df_summary.to_string(index=False))

    # ── Tests de Wilcoxon ──
    print("\n>>> Ejecutando tests de Wilcoxon pareados...")
    df_wilcoxon = run_wilcoxon_tests(df_results)
    df_wilcoxon.to_csv('wilcoxon_tests.csv', index=False)
    print_wilcoxon_summary(df_wilcoxon)

    # ── Figuras ──
    print("\n>>> Generando figuras para el TFM...")
    plot_rmse_interaction(df_results)
    plot_me_bias(df_results)
    plot_nrmse_by_method(df_results)
    plot_corr_preservation(df_results)
    plot_loss_curves(all_loss_curves)
    plot_umap_comparison(df_results)  # Al final porque tarda más
    print("\nNota: proyección UMAP generada con n_neighbors=15, min_dist=0.1 (UMAP 0.5+)")

    # ── Análisis de reclasificación competencial ──
    print("\n>>> Iniciando análisis de reclasificación competencial...")
    print("    Condiciones: MAR 30% y MNAR 30% | Escenario simple (6 factores)")
    df_reclasif = run_reclassification_analysis(
        n_factors=6,
        items_per_factor=5,
        mechanisms_target=['MAR', 'MNAR'],
        missing_rate=0.30,
        n_reps=N_REPETITIONS,
    )
    df_reclasif.to_csv('reclasificacion_cruda.csv', index=False)
    print("\n>>> Tabla de reclasificación competencial:")
    generate_reclassification_table(df_reclasif)

    print("\n" + "=" * 60)
    print("SIMULACIÓN COMPLETADA")
    print("Archivos generados:")
    print("  📄 resultados_crudos.csv")
    print("  📄 tabla_resumen_resultados.csv")
    print("  📄 wilcoxon_tests.csv")
    print("  📄 reclasificacion_cruda.csv")
    print("  📄 tabla_reclasificacion_competencial.csv")
    print("  🖼️  figura1_rmse_interaccion.png")
    print("  🖼️  figura2_sesgo_medio_me.png")
    print("  🖼️  figura3_nrmse_metodos.png")
    print("  🖼️  figura4_preservacion_correlaciones.png")
    print("  🖼️  figura5_curva_loss_autoencoder.png")
    print("  🖼️  figura6_umap_comparacion.png")
    print("=" * 60)
