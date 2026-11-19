# -*- coding: utf-8 -*-
"""
===============================================================================
 Validación cuantitativa del modelo del bot XRP/USDT (velas de 2 minutos)
 Proyecto Final Integrador · Finanzas Computacionales, Algoritmos Cuantitativos e IA
===============================================================================

Qué hace (Fase 4 de la rúbrica):
  1. Descarga velas reales de 1 min de XRP_USDT (futuros perpetuos, API pública de Gate.io)
     y arma velas de 2 min cerradas, igual que el bot.
  2. Construye las MISMAS variables que usa el bot (EMA 9/21, RSI 14, rendimientos, volatilidad).
  3. Clasificación (Buy/Sell = sube/baja en 3 velas):
        Random Forest vs KNN vs modelo base  ->  Matriz de confusión, Accuracy, Precision,
        Recall, F1-Score.
  4. Regresión (precio dentro de 3 velas):
        Random Forest vs KNN vs modelo ingenuo (random walk: precio futuro = precio actual)
        ->  RMSE, MSE, MAE, MAPE, SMAPE, Variance Score.
  5. Test de Diebold-Mariano (con corrección Harvey-Leybourne-Newbold) para comparar
     estadísticamente los errores de pronóstico entre modelos.
  6. Simulación de hiperparámetros del Random Forest (n_estimators x max_depth) con su
     tabla comparativa y mapa de calor.

Uso:   python validacion_rf.py            (necesita internet)
Salida: carpeta  resultados/  con CSVs, gráficas PNG y resumen_validacion.md
===============================================================================
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, confusion_matrix, explained_variance_score,
                             f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, recall_score)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Parámetros (los mismos que el bot)
# -----------------------------------------------------------------------------
CONTRATO = "XRP_USDT"
URL_VELAS = "https://api.gateio.ws/api/v4/futures/usdt/candlesticks"
MINUTOS_HISTORIA = 9900            # Gate.io solo sirve las últimas 10,000 velas de 1 min (~6.9 días)
TEMPORALIDAD_MIN = 2
HORIZONTE = 3                      # velas de 2 min hacia adelante (6 minutos), igual que el bot
PORCENTAJE_TEST = 0.20
EMA_FAST, EMA_SLOW, RSI_PERIOD = 9, 21, 14
RF_BOT = dict(n_estimators=200, max_depth=6, min_samples_leaf=20)   # configuración en producción
GRID_ARBOLES = [50, 100, 200, 400]
GRID_PROFUNDIDAD = [4, 6, 8, None]
KNN_VECINOS = 25
SEMILLA = 42

SALIDA = Path(__file__).resolve().parent / "resultados"
SALIDA.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# 1. Datos
# -----------------------------------------------------------------------------
def descargar_velas_1m(minutos=MINUTOS_HISTORIA):
    """Velas de 1 min de la API pública de Gate.io, en bloques de 900 minutos."""
    fin = int(time.time()) // 60 * 60
    inicio = fin - minutos * 60
    filas, t = [], inicio
    while t < fin:
        t_fin = min(t + 900 * 60, fin)
        r = requests.get(URL_VELAS, timeout=30, params={
            "contract": CONTRATO, "interval": "1m", "from": t, "to": t_fin})
        r.raise_for_status()
        filas += r.json()
        t = t_fin
        time.sleep(0.15)
    df = pd.DataFrame([{"t": int(b["t"]), "open": float(b["o"]), "high": float(b["h"]),
                        "low": float(b["l"]), "close": float(b["c"]), "vol": float(b.get("v") or 0)}
                       for b in filas])
    df = df.drop_duplicates("t").sort_values("t")
    return df[df.t + 60 <= time.time()].reset_index(drop=True)


def velas_2m(df1):
    seg = TEMPORALIDAD_MIN * 60
    df1 = df1.assign(grupo=df1.t - df1.t % seg)
    g = df1.groupby("grupo")
    v = pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(),
                      "close": g.close.last(), "vol": g.vol.sum(), "n": g.t.count()})
    v = v[v.n == TEMPORALIDAD_MIN].drop(columns="n")
    v.index = pd.to_datetime(v.index, unit="s", utc=True)
    return v


def datos_sinteticos(n=4950):
    """Solo para probar el script sin internet (NO se usa para los resultados)."""
    rng = np.random.default_rng(SEMILLA)
    c = 1.0 * np.exp(np.cumsum(rng.normal(0, 0.0012, n)))
    idx = pd.date_range("2026-11-01", periods=n, freq="2min", tz="UTC")
    return pd.DataFrame({"open": np.r_[c[0], c[:-1]], "high": c * 1.0008, "low": c * 0.9992,
                         "close": c, "vol": rng.uniform(1e4, 5e4, n)}, index=idx)


# -----------------------------------------------------------------------------
# 2. Variables (idénticas al bot)
# -----------------------------------------------------------------------------
FEATURES = ["ret_1", "ret_2", "ret_3", "ret_6", "ret_12", "ema_gap", "rsi", "vol_12", "rango"]


def variables(df):
    df = df.copy()
    c = df.close
    df["ema_fast"] = c.ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = c.ewm(span=EMA_SLOW, adjust=False).mean()
    d = c.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    df["rsi"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(50)
    df["ret"] = c.pct_change()
    for n in (1, 2, 3, 6, 12):
        df[f"ret_{n}"] = c.pct_change(n)
    df["ema_gap"] = df.ema_fast / df.ema_slow - 1
    df["vol_12"] = df.ret.rolling(12).std()
    df["rango"] = (df.high - df.low) / c
    df["precio_futuro"] = c.shift(-HORIZONTE)
    df["sube"] = (df.precio_futuro > c).astype(int)
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["precio_futuro"])


def dividir(df):
    """Partición temporal (sin mezclar) con purga de HORIZONTE velas para evitar fuga de información."""
    corte = int(len(df) * (1 - PORCENTAJE_TEST))
    return df.iloc[:corte - HORIZONTE], df.iloc[corte:]


# -----------------------------------------------------------------------------
# 3. Métricas
# -----------------------------------------------------------------------------
def smape(y, p):
    return float(np.mean(2 * np.abs(p - y) / (np.abs(y) + np.abs(p))) * 100)


def mape(y, p):
    return float(np.mean(np.abs((y - p) / y)) * 100)


def metricas_regresion(y, p):
    mse = mean_squared_error(y, p)
    return {"RMSE": float(np.sqrt(mse)), "MSE": float(mse), "MAE": float(mean_absolute_error(y, p)),
            "MAPE (%)": mape(y, p), "SMAPE (%)": smape(y, p),
            "Variance Score": float(explained_variance_score(y, p))}


def metricas_clasificacion(y, p):
    return {"Accuracy": accuracy_score(y, p), "Precision": precision_score(y, p, zero_division=0),
            "Recall": recall_score(y, p, zero_division=0), "F1-Score": f1_score(y, p, zero_division=0)}


def diebold_mariano(e1, e2, h=HORIZONTE, potencia=2):
    """Test DM (H0: igual precisión) con corrección HLN. d_t = |e1|^p - |e2|^p.
    DM < 0 -> el modelo 1 tiene menor error que el modelo 2."""
    d = np.abs(e1) ** potencia - np.abs(e2) ** potencia
    n = len(d)
    d_media = d.mean()
    autocov = [np.mean((d[k:] - d_media) * (d[:n - k] - d_media)) for k in range(h)]
    var_d = (autocov[0] + 2 * sum(autocov[1:])) / n
    dm = d_media / np.sqrt(var_d)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * hln
    p = 2 * stats.t.cdf(-abs(dm_hln), df=n - 1)
    return float(dm_hln), float(p)


# -----------------------------------------------------------------------------
# 4. Experimentos
# -----------------------------------------------------------------------------
def main(sintetico=False):
    print(">> Descargando datos..." if not sintetico else ">> Datos SINTÉTICOS (solo prueba)")
    v2 = datos_sinteticos() if sintetico else velas_2m(descargar_velas_1m())
    df = variables(v2)
    train, test = dividir(df)
    Xtr, Xte = train[FEATURES], test[FEATURES]
    fuente = "sintética (prueba)" if sintetico else f"Gate.io API pública, futuros {CONTRATO}"
    print(f"   {len(df)} velas de 2 min · train {len(train)} · test {len(test)} · "
          f"{df.index[0]:%Y-%m-%d %H:%M} → {df.index[-1]:%Y-%m-%d %H:%M} UTC")

    # ---------- Clasificación: ¿sube en 3 velas? ----------
    ytr, yte = train.sube, test.sube
    rf_c = RandomForestClassifier(**RF_BOT, random_state=SEMILLA, n_jobs=-1).fit(Xtr, ytr)
    knn_c = make_pipeline(StandardScaler(), KNeighborsClassifier(KNN_VECINOS)).fit(Xtr, ytr)
    base_c = (test.ret_1 > 0).astype(int)          # modelo base: momentum de la última vela
    pred_c = {"Random Forest (bot)": rf_c.predict(Xte), "KNN": knn_c.predict(Xte), "Base (momentum)": base_c}
    tabla_c = pd.DataFrame({m: metricas_clasificacion(yte, p) for m, p in pred_c.items()}).T.round(4)
    tabla_c.to_csv(SALIDA / "metricas_clasificacion.csv")

    pd.DataFrame([{"modelo": m, "TN": cm[0, 0], "FP": cm[0, 1], "FN": cm[1, 0], "TP": cm[1, 1]}
                  for m, cm in ((m, confusion_matrix(yte, p)) for m, p in pred_c.items())]
                 ).to_csv(SALIDA / "matrices_confusion.csv", index=False)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (m, p) in zip(axs, pred_c.items()):
        cm = confusion_matrix(yte, p)
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                        color="white" if cm[i, j] > (cm.max() + cm.min()) / 2 else "black")
        ax.set_xticks([0, 1], ["Baja/Sell", "Sube/Buy"])
        ax.set_yticks([0, 1], ["Baja/Sell", "Sube/Buy"])
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.set_title(f"{m}\nF1 = {f1_score(yte, p):.3f}")
    fig.suptitle(f"Matrices de confusión · XRP/USDT 2 min · horizonte {HORIZONTE} velas · test n={len(test)}")
    fig.tight_layout()
    fig.savefig(SALIDA / "matrices_confusion.png", dpi=130)
    plt.close(fig)

    # ---------- Regresión: precio dentro de 3 velas ----------
    ytr_r, yte_r = train.precio_futuro, test.precio_futuro
    # Se modela el rendimiento futuro y se reconstruye el precio (más estable que el nivel)
    rtr = ytr_r / train.close - 1
    rf_r = RandomForestRegressor(**RF_BOT, random_state=SEMILLA, n_jobs=-1).fit(Xtr, rtr)
    knn_r = make_pipeline(StandardScaler(), KNeighborsRegressor(KNN_VECINOS)).fit(Xtr, rtr)
    pred_r = {"Random Forest (bot)": test.close * (1 + rf_r.predict(Xte)),
              "KNN": test.close * (1 + knn_r.predict(Xte)),
              "Ingenuo (random walk)": test.close.values}
    tabla_r = pd.DataFrame({m: metricas_regresion(yte_r.values, np.asarray(p)) for m, p in pred_r.items()}).T
    tabla_r.to_csv(SALIDA / "metricas_regresion.csv")

    pd.DataFrame({"precio_actual": test.close.values, "precio_real_t3": yte_r.values,
                  "pred_rf": np.asarray(pred_r["Random Forest (bot)"]), "pred_knn": np.asarray(pred_r["KNN"]),
                  "real_sube": yte.values, "pred_rf_sube": pred_c["Random Forest (bot)"],
                  "prob_rf_sube": rf_c.predict_proba(Xte)[:, 1]},
                 index=test.index).to_csv(SALIDA / "predicciones_test.csv", float_format="%.6f")

    # ---------- Diebold-Mariano ----------
    err = {m: yte_r.values - np.asarray(p) for m, p in pred_r.items()}
    pares = [("Random Forest (bot)", "Ingenuo (random walk)"), ("Random Forest (bot)", "KNN"),
             ("KNN", "Ingenuo (random walk)")]
    dm_filas = []
    for a, b in pares:
        dm, p = diebold_mariano(err[a], err[b])
        mejor = a if dm < 0 else b
        dm_filas.append({"Modelo 1": a, "Modelo 2": b, "DM (HLN)": round(dm, 4), "p-valor": round(p, 4),
                         "Conclusión (α=5%)": (f"{mejor} tiene menor error (significativo)" if p < 0.05
                                              else "Sin diferencia significativa")})
    tabla_dm = pd.DataFrame(dm_filas)
    tabla_dm.to_csv(SALIDA / "diebold_mariano.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 4))
    n_plot = min(300, len(test))
    ax.plot(test.index[-n_plot:], yte_r.values[-n_plot:], color="black", lw=1.2, label="Precio real (t+3)")
    for m, colr in [("Random Forest (bot)", "tab:blue"), ("KNN", "tab:orange")]:
        ax.plot(test.index[-n_plot:], np.asarray(pred_r[m])[-n_plot:], lw=0.9, alpha=0.8, color=colr, label=m)
    ax.set_title("Pronóstico del precio a 6 minutos vs real (últimas 300 velas de prueba)")
    ax.set_ylabel("XRP/USDT")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SALIDA / "pronostico_vs_real.png", dpi=130)
    plt.close(fig)

    # ---------- Simulación de hiperparámetros ----------
    filas = []
    for n in GRID_ARBOLES:
        for prof in GRID_PROFUNDIDAD:
            t0 = time.time()
            m_r = RandomForestRegressor(n_estimators=n, max_depth=prof, min_samples_leaf=20,
                                        random_state=SEMILLA, n_jobs=-1).fit(Xtr, rtr)
            p_r = test.close.values * (1 + m_r.predict(Xte))
            m_c = RandomForestClassifier(n_estimators=n, max_depth=prof, min_samples_leaf=20,
                                         random_state=SEMILLA, n_jobs=-1).fit(Xtr, ytr)
            dm, pv = diebold_mariano(yte_r.values - p_r, err["Ingenuo (random walk)"])
            fila = {"n_estimators": n, "max_depth": str(prof),
                    **metricas_regresion(yte_r.values, p_r),
                    "F1-Score": f1_score(yte, m_c.predict(Xte)),
                    "Accuracy": accuracy_score(yte, m_c.predict(Xte)),
                    "DM vs ingenuo": dm, "p-valor DM": pv, "segundos": time.time() - t0}
            filas.append(fila)
            print(f"   RF n={n:>3} prof={str(prof):>4}  RMSE={fila['RMSE']:.6f}  F1={fila['F1-Score']:.3f}")
    tabla_h = pd.DataFrame(filas)
    tabla_h.to_csv(SALIDA / "simulacion_hiperparametros.csv", index=False)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, col, titulo, cmap in [(axs[0], "RMSE", "RMSE (menor es mejor)", "Blues_r"),
                                  (axs[1], "F1-Score", "F1-Score (mayor es mejor)", "Greens")]:
        piv = tabla_h.pivot(index="n_estimators", columns="max_depth", values=col)
        piv = piv[[str(c) for c in GRID_PROFUNDIDAD]]
        im = ax.imshow(piv.values, cmap=cmap, aspect="auto")
        ax.set_xticks(range(piv.shape[1]), piv.columns)
        ax.set_yticks(range(piv.shape[0]), piv.index)
        ax.set_xlabel("max_depth")
        ax.set_ylabel("n_estimators")
        ax.set_title(titulo)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.5f}" if col == "RMSE" else f"{piv.values[i, j]:.3f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if ((piv.values[i, j] - piv.values.min()) / (np.ptp(piv.values) or 1)
                                         > 0.5) != cmap.endswith("_r") else "black")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Simulación de hiperparámetros del Random Forest")
    fig.tight_layout()
    fig.savefig(SALIDA / "hiperparametros.png", dpi=130)
    plt.close(fig)

    # ---------- Resumen en Markdown ----------
    mejor = tabla_h.sort_values("RMSE").iloc[0]
    def md(t, index=True):
        return t.to_markdown(index=index, floatfmt=".6g")
    resumen = f"""# Resultados de validación del modelo

- **Datos:** {fuente} · {len(df):,} velas de 2 min · {df.index[0]:%Y-%m-%d %H:%M} → {df.index[-1]:%Y-%m-%d %H:%M} UTC
- **Partición temporal:** train {len(train):,} / test {len(test):,} (último {int(PORCENTAJE_TEST*100)} %), con purga de {HORIZONTE} velas
- **Objetivo:** dirección (clasificación) y precio (regresión) a {HORIZONTE} velas = {HORIZONTE*TEMPORALIDAD_MIN} minutos
- **Random Forest del bot:** {RF_BOT} · KNN con k={KNN_VECINOS} (variables estandarizadas)

## Clasificación (Buy/Sell)
{md(tabla_c)}

![Matrices de confusión](matrices_confusion.png)

## Regresión (precio a {HORIZONTE*TEMPORALIDAD_MIN} min)
{md(tabla_r)}

![Pronóstico vs real](pronostico_vs_real.png)

## Test de Diebold-Mariano (pérdida cuadrática, corrección HLN, h={HORIZONTE})
{md(tabla_dm, index=False)}

DM < 0 indica que el **Modelo 1** tiene menor error cuadrático que el Modelo 2.

## Simulación de hiperparámetros
{md(tabla_h.round(6), index=False)}

Mejor combinación por RMSE: **n_estimators={mejor['n_estimators']}, max_depth={mejor['max_depth']}**
(RMSE {mejor['RMSE']:.6g}, F1 {mejor['F1-Score']:.3f}).

![Hiperparámetros](hiperparametros.png)
"""
    (SALIDA / "resumen_validacion.md").write_text(resumen, encoding="utf-8")
    (SALIDA / "metadatos.json").write_text(json.dumps({
        "fuente": fuente, "velas": len(df), "inicio_utc": str(df.index[0]), "fin_utc": str(df.index[-1]),
        "train": len(train), "test": len(test), "horizonte_velas": HORIZONTE}, indent=2), encoding="utf-8")
    print("\n" + tabla_c.to_string())
    print("\n" + tabla_r.to_string())
    print("\n" + tabla_dm.to_string(index=False))
    import shutil
    zip_path = shutil.make_archive(str(SALIDA.parent / "resultados_validacion"), "zip", SALIDA)
    print(f"\n>> Listo. Resultados en: {SALIDA}  ·  ZIP: {zip_path}")


if __name__ == "__main__":
    main(sintetico="--sintetico" in sys.argv)
