# -*- coding: utf-8 -*-
"""
Regenera las gráficas y el resumen de validación a partir de los CSV de resultados/
(sin volver a descargar datos ni reentrenar). Útil para el reporte y la presentación.

Uso:  python graficas_validacion.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent / "resultados"
AZUL, NARANJA, VERDE, GRIS = "#2a6fdb", "#e8833a", "#2e9d5b", "#555555"


def matrices():
    m = pd.read_csv(R / "matrices_confusion.csv")
    f1 = pd.read_csv(R / "metricas_clasificacion.csv", index_col=0)["F1-Score"]
    fig, axs = plt.subplots(1, len(m), figsize=(13, 4))
    for ax, (_, r) in zip(axs, m.iterrows()):
        cm = np.array([[r.TN, r.FP], [r.FN, r.TP]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                        color="white" if cm[i, j] > (cm.max() + cm.min()) / 2 else "black")
        ax.set_xticks([0, 1], ["Baja/Sell", "Sube/Buy"])
        ax.set_yticks([0, 1], ["Baja/Sell", "Sube/Buy"])
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.set_title(f"{r.modelo}\nF1 = {f1[r.modelo]:.3f}")
    n = int(m.iloc[0][["TN", "FP", "FN", "TP"]].sum())
    fig.suptitle(f"Matrices de confusión · XRP/USDT 2 min · horizonte 3 velas · test n={n}")
    fig.tight_layout()
    fig.savefig(R / "matrices_confusion.png", dpi=130)
    plt.close(fig)


def pronostico():
    archivo = R / "predicciones_test.csv"
    if not archivo.exists():
        archivo = R / "predicciones_test_ultimas120.csv"
    p = pd.read_csv(archivo, index_col=0, parse_dates=True).tail(120)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(p.index, p.precio_real_t3, color="black", lw=1.4, label="Precio real (t+3)")
    ax.plot(p.index, p.pred_rf, color=AZUL, lw=1, label="Random Forest (bot)")
    ax.plot(p.index, p.pred_knn, color=NARANJA, lw=1, alpha=0.85, label="KNN")
    ax.set_title("Pronóstico del precio a 6 minutos vs real (últimas 120 velas de prueba, UTC)")
    ax.set_ylabel("XRP/USDT")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(R / "pronostico_vs_real.png", dpi=130)
    plt.close(fig)


def comparativa():
    reg = pd.read_csv(R / "metricas_regresion.csv", index_col=0)
    dm = pd.read_csv(R / "diebold_mariano.csv")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    colores = [AZUL, NARANJA, GRIS]
    for ax, col in zip(axs, ["RMSE", "MAE"]):
        ax.bar(reg.index, reg[col], color=colores)
        ax.set_ylim(reg[col].min() * 0.97, reg[col].max() * 1.02)
        ax.set_title(f"{col} del precio a 6 min (menor es mejor)")
        for i, v in enumerate(reg[col]):
            ax.text(i, v, f"{v:.6f}", ha="center", va="bottom", fontsize=9)
        ax.tick_params(axis="x", labelsize=9)
    rf_rw = dm.iloc[0]
    fig.suptitle(f"Comparativa de modelos · Diebold-Mariano RF vs random walk: "
                 f"DM={rf_rw['DM (HLN)']:.2f}, p={rf_rw['p-valor']:.3f}")
    fig.tight_layout()
    fig.savefig(R / "comparativa_modelos.png", dpi=130)
    plt.close(fig)


def hiperparametros():
    h = pd.read_csv(R / "simulacion_hiperparametros.csv", dtype={"max_depth": str}, keep_default_na=False)
    h["max_depth"] = h["max_depth"].astype(str)
    orden = ["4", "6", "8", "None"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, col, titulo, cmap in [(axs[0], "RMSE", "RMSE (menor es mejor)", "Blues_r"),
                                  (axs[1], "F1-Score", "F1-Score (mayor es mejor)", "Greens")]:
        piv = h.pivot(index="n_estimators", columns="max_depth", values=col)[orden]
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
    fig.suptitle("Simulación de hiperparámetros del Random Forest (min_samples_leaf=20)")
    fig.tight_layout()
    fig.savefig(R / "hiperparametros.png", dpi=130)
    plt.close(fig)


def resumen():
    meta = json.loads((R / "metadatos.json").read_text(encoding="utf-8"))
    c = pd.read_csv(R / "metricas_clasificacion.csv", index_col=0)
    reg = pd.read_csv(R / "metricas_regresion.csv", index_col=0)
    dm = pd.read_csv(R / "diebold_mariano.csv")
    h = pd.read_csv(R / "simulacion_hiperparametros.csv", dtype={"max_depth": str}, keep_default_na=False)
    mejor = h.sort_values("RMSE").iloc[0]
    texto = f"""# Resultados de validación del modelo

- **Datos:** {meta['fuente']} · {meta['velas']:,} velas de 2 min · {meta['inicio_utc'][:16]} → {meta['fin_utc'][:16]} UTC
- **Partición temporal:** train {meta['train']:,} / test {meta['test']:,} (último 20 %), con purga de {meta['horizonte_velas']} velas
- **Objetivo:** dirección (clasificación) y precio (regresión) a {meta['horizonte_velas']} velas = 6 minutos

## Clasificación (Buy/Sell)
{c.to_markdown(floatfmt='.4f')}

![Matrices de confusión](matrices_confusion.png)

## Regresión (precio a 6 min)
{reg.to_markdown(floatfmt='.6g')}

![Comparativa](comparativa_modelos.png)

![Pronóstico vs real](pronostico_vs_real.png)

## Test de Diebold-Mariano (pérdida cuadrática, corrección HLN, h=3)
{dm.to_markdown(index=False)}

DM < 0 indica que el **Modelo 1** tiene menor error cuadrático que el Modelo 2.

## Simulación de hiperparámetros
{h.drop(columns=['segundos']).to_markdown(index=False, floatfmt='.6g')}

Mejor combinación por RMSE: **n_estimators={int(mejor['n_estimators'])}, max_depth={mejor['max_depth']}**
(RMSE {mejor['RMSE']:.6g}, F1 {mejor['F1-Score']:.3f}).

![Hiperparámetros](hiperparametros.png)
"""
    (R / "resumen_validacion.md").write_text(texto, encoding="utf-8")


if __name__ == "__main__":
    matrices()
    pronostico()
    comparativa()
    hiperparametros()
    resumen()
    print("Gráficas y resumen regenerados en", R)
