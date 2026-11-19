# -*- coding: utf-8 -*-
"""
Análisis de resultados en Gate.io TESTNET (Fase 6 de la rúbrica).

Lee el Excel que genera el bot (~/bot_xrp_datos/operaciones_xrp.xlsx) y produce:
  - Métricas de trading: Win Rate, PnL neto, profit factor, drawdown, ROI simulado.
  - Matriz de confusión de las señales ejecutadas: dirección operada (LONG = Sube, SHORT = Baja)
    contra la dirección real del precio entre la entrada y la salida, más Accuracy/Precision/Recall/F1.
  - Gráficas: curva de capital (PnL acumulado) y matriz de confusión.

Uso:   python analisis_testnet.py  [ruta_excel]
Salida: resultados/testnet_*.csv / .png / .md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NOTIONAL_USDT = 100      # tamaño de cada operación en el bot
LEVERAGE = 1             # apalancamiento del bot -> margen = nocional
EXCEL = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "bot_xrp_datos" / "operaciones_xrp.xlsx"
SALIDA = Path(__file__).resolve().parent / "resultados"
SALIDA.mkdir(exist_ok=True)


def main():
    ops = pd.read_excel(EXCEL, sheet_name="Operaciones")
    if ops.empty:
        print("Todavía no hay operaciones cerradas.")
        return
    pnl = ops["PnL (USDT)"]
    gan, per = pnl[pnl > 0].sum(), -pnl[pnl <= 0].sum()
    curva = pnl.cumsum()
    margen = NOTIONAL_USDT / LEVERAGE
    metricas = {
        "Operaciones cerradas": len(ops),
        "Ganadoras": int((pnl > 0).sum()),
        "Perdedoras": int((pnl <= 0).sum()),
        "Win Rate (%)": round((pnl > 0).mean() * 100, 2),
        "PnL neto (USDT)": round(pnl.sum(), 4),
        "Ganancia promedio (USDT)": round(pnl[pnl > 0].mean(), 4) if (pnl > 0).any() else 0,
        "Pérdida promedio (USDT)": round(pnl[pnl <= 0].mean(), 4) if (pnl <= 0).any() else 0,
        "Profit factor": round(gan / per, 3) if per > 0 else np.nan,
        "Máximo drawdown (USDT)": round(float((curva - curva.cummax().clip(lower=0)).min()), 4),
        "ROI simulado por operación (%)": round(pnl.mean() / margen * 100, 4),
        "ROI simulado acumulado (%)": round(pnl.sum() / margen * 100, 4),
    }
    pd.Series(metricas).to_csv(SALIDA / "testnet_metricas.csv", header=["valor"])

    # ---- Matriz de confusión de la dirección operada ----
    pred = (ops["Posición"].str.upper() == "LONG").astype(int)                 # 1 = apostó a que sube
    real = (ops["Precio salida"] > ops["Precio entrada"]).astype(int)          # 1 = el precio subió
    cm = confusion_matrix(real, pred, labels=[0, 1])
    clas = {"Accuracy": accuracy_score(real, pred),
            "Precision": precision_score(real, pred, zero_division=0),
            "Recall": recall_score(real, pred, zero_division=0),
            "F1-Score": f1_score(real, pred, zero_division=0)}
    pd.Series(clas).round(4).to_csv(SALIDA / "testnet_clasificacion.csv", header=["valor"])

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(1, len(ops) + 1)
    axs[0].bar(x, pnl, color=np.where(pnl > 0, "#2e9d5b", "#d1453b"), alpha=0.6, label="PnL por operación")
    axs[0].plot(x, curva, color="#2a6fdb", marker="o", lw=2, label="PnL acumulado")
    axs[0].axhline(0, color="black", lw=0.8)
    axs[0].set_xlabel("Operación #")
    axs[0].set_ylabel("USDT")
    axs[0].set_title(f"Curva de capital Testnet · Win Rate {metricas['Win Rate (%)']}% · "
                     f"PnL {metricas['PnL neto (USDT)']:+.3f} USDT")
    axs[0].legend()
    axs[1].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axs[1].text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                        color="white" if cm[i, j] > (cm.max() + cm.min()) / 2 else "black")
    axs[1].set_xticks([0, 1], ["SHORT (baja)", "LONG (sube)"])
    axs[1].set_yticks([0, 1], ["Bajó", "Subió"])
    axs[1].set_xlabel("Señal ejecutada")
    axs[1].set_ylabel("Movimiento real")
    axs[1].set_title(f"Matriz de confusión de señales · F1 {clas['F1-Score']:.3f}")
    fig.tight_layout()
    fig.savefig(SALIDA / "testnet_resultados.png", dpi=130)
    plt.close(fig)

    cols = ["#", "Fecha entrada", "Posición", "Precio entrada", "Precio salida", "PnL (USDT)",
            "Motivo salida", "RF P(sube)", "VaR95 %", "Gemini"]
    tabla = ops[[c for c in cols if c in ops.columns]]
    tabla.to_csv(SALIDA / "testnet_operaciones.csv", index=False)
    (SALIDA / "testnet_resumen.md").write_text(
        "# Resultados en Gate.io Testnet\n\n"
        + pd.Series(metricas).to_frame("valor").to_markdown() + "\n\n"
        + "## Clasificación de señales\n\n" + pd.Series(clas).round(4).to_frame("valor").to_markdown()
        + "\n\n![Resultados](testnet_resultados.png)\n\n## Operaciones\n\n"
        + tabla.to_markdown(index=False) + "\n", encoding="utf-8")
    print(pd.Series(metricas).to_string())
    print(pd.Series(clas).round(4).to_string())


if __name__ == "__main__":
    main()
