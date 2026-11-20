# -*- coding: utf-8 -*-
"""Genera docs/Reporte_Tecnico_Bot_XRP.pdf a partir de los resultados del repositorio."""
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

RAIZ = Path(__file__).resolve().parent.parent
RES = RAIZ / "resultados"
SALIDA = RAIZ / "docs" / "Reporte_Tecnico_Bot_XRP.pdf"
REPO = "https://github.com/caste90210/bot-xrp-trading"
APP = "https://remix-risksignal-ai-1695.ai.studio"

F = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("DV", F + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DVB", F + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DVI", F + "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFont(TTFont("DVM", F + "DejaVuSansMono.ttf"))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI", boldItalic="DVB")

AZUL = colors.HexColor("#1f4e9c")
GRIS = colors.HexColor("#f2f4f8")
ss = getSampleStyleSheet()
N = ParagraphStyle("N", parent=ss["Normal"], fontName="DV", fontSize=9.6, leading=13.6, alignment=TA_JUSTIFY,
                   spaceAfter=5)
H1 = ParagraphStyle("H1", parent=N, fontName="DVB", fontSize=14, leading=18, textColor=AZUL, spaceBefore=10,
                    spaceAfter=6, alignment=0)
H2 = ParagraphStyle("H2", parent=N, fontName="DVB", fontSize=11, leading=14, textColor=AZUL, spaceBefore=7,
                    spaceAfter=3, alignment=0)
EQ = ParagraphStyle("EQ", parent=N, fontName="DV", alignment=TA_CENTER, backColor=GRIS, borderPadding=5,
                    spaceBefore=4, spaceAfter=8)
CAP = ParagraphStyle("CAP", parent=N, fontName="DVI", fontSize=8.3, leading=11, alignment=TA_CENTER,
                     textColor=colors.HexColor("#555555"))
CELL = ParagraphStyle("CELL", parent=N, fontSize=8, leading=10, alignment=0, spaceAfter=0)
CELLB = ParagraphStyle("CELLB", parent=CELL, fontName="DVB", textColor=colors.white)
BUL = ParagraphStyle("BUL", parent=N, leftIndent=12, bulletIndent=2)


def p(t, s=N):
    return Paragraph(t, s)


def bullets(items):
    return [Paragraph(t, BUL, bulletText="•") for t in items]


def tabla(filas, anchos, cab=True, fs=None):
    c1, c2 = CELL, CELLB
    if fs:
        c1 = ParagraphStyle("c1", parent=CELL, fontSize=fs, leading=fs + 2)
        c2 = ParagraphStyle("c2", parent=CELLB, fontSize=fs, leading=fs + 2)
    data = [[Paragraph(str(c), c2 if (cab and i == 0) else c1) for c in fila] for i, fila in enumerate(filas)]
    t = Table(data, colWidths=anchos, repeatRows=1 if cab else 0)
    estilo = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c5cbd6")),
              ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
              ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GRIS]),
              ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if cab:
        estilo.append(("BACKGROUND", (0, 0), (-1, 0), AZUL))
    t.setStyle(TableStyle(estilo))
    return t


def figura(ruta, ancho_cm, pie):
    from PIL import Image as PI
    w, h = PI.open(ruta).size
    img = Image(str(ruta), width=ancho_cm * cm, height=ancho_cm * cm * h / w)
    return KeepTogether([img, Spacer(1, 2), p(pie, CAP), Spacer(1, 6)])


def pie_pagina(c, d):
    c.saveState()
    c.setFont("DV", 7.5)
    c.setFillColor(colors.HexColor("#777777"))
    c.drawString(2 * cm, 1.2 * cm, "Proyecto Final Integrador · Bot de trading XRP/USDT con IA")
    c.drawRightString(letter[0] - 2 * cm, 1.2 * cm, f"Página {d.page}")
    c.restoreState()


def main():
    meta = pd.read_json(RES / "metadatos.json", typ="series")
    clas = pd.read_csv(RES / "metricas_clasificacion.csv", index_col=0)
    reg = pd.read_csv(RES / "metricas_regresion.csv", index_col=0)
    dm = pd.read_csv(RES / "diebold_mariano.csv")
    hip = pd.read_csv(RES / "simulacion_hiperparametros.csv", dtype={"max_depth": str}, keep_default_na=False)
    tn = pd.read_csv(RES / "testnet_metricas.csv", index_col=0)["valor"]
    tops = pd.read_csv(RES / "testnet_operaciones.csv")

    s = []
    # ---------------- Portada ----------------
    s += [Spacer(1, 8.5 * cm),
          p("Proyecto Integrador<br/>Finanzas Computacionales",
            ParagraphStyle("T", parent=H1, fontSize=30, leading=40, alignment=TA_CENTER)),
          PageBreak()]

    # ---------------- Resumen ----------------
    s += [p("Resumen ejecutivo", H1),
          p("Se desarrolló un bot de trading para futuros perpetuos <b>XRP_USDT</b> en <b>Gate.io Testnet</b> que opera "
            "con velas de 2 minutos. La señal combina tendencia (EMA 9/21), momentum (RSI 14) y la probabilidad de "
            "subida estimada por un <b>Random Forest</b>; cada señal la valida un agente <b>Gemini</b> que ajusta el "
            "Stop Loss y el Take Profit, ambos calculados como múltiplos del <b>VaR histórico al 95 %</b>. El bot corre "
            "24/7 como servicio <i>systemd</i> en una VM <b>e2-small</b> de Google Cloud, se controla desde "
            "<b>Telegram</b> y publica su estado y operaciones en <b>Firebase Firestore</b>, desde donde la app "
            "<b>RiskSignal AI</b> (Google AI Studio) los muestra en tiempo real."),
          p(f"La validación con {int(meta['velas']):,} velas reales de 2 minutos ({str(meta['inicio_utc'])[:10]} a "
            f"{str(meta['fin_utc'])[:10]}) muestra que, a un horizonte de 6 minutos, el Random Forest no supera "
            f"estadísticamente a un modelo ingenuo de caminata aleatoria (Diebold-Mariano, DM = "
            f"{dm.iloc[0]['DM (HLN)']:.2f}, p = {dm.iloc[0]['p-valor']:.3f}), resultado consistente con la hipótesis "
            "de mercados eficientes en su forma débil. Por ello el modelo se emplea como <b>filtro de confianza</b> "
            "(probabilidad ≥ 0.55) dentro de un sistema de reglas con gestión de riesgo, y no como señal única. La "
            "simulación de 16 combinaciones de hiperparámetros evidencia el compromiso entre error de pronóstico "
            "(menor con árboles poco profundos) y capacidad de detectar movimientos (mayor F1 con árboles profundos)."),
          p("Objetivos", H2)]
    s += bullets(["Construir un bot autónomo que integre análisis técnico, aprendizaje automático y medición de riesgo.",
                  "Desplegarlo de forma continua en la nube con secretos fuera del código y bitácora persistente.",
                  "Integrarlo con una aplicación de IA para monitoreo en tiempo real.",
                  "Validar el modelo con métricas de regresión y clasificación y con la prueba de Diebold-Mariano."])

    # ---------------- Arquitectura ----------------
    s += [p("1. Arquitectura del sistema", H1),
          figura(RAIZ / "docs" / "arquitectura.png", 16.5, "Figura 1. Arquitectura: VM en Google Cloud, exchange, "
                 "agente Gemini, Telegram, Firestore y la app RiskSignal AI."),
          p("Ciclo por vela (cada 2 minutos, revisión de precio cada 5 s):", H2)]
    s += bullets(["Descarga velas de 1 minuto y construye la vela de 2 minutos <b>cerrada</b> (sin usar la vela en curso).",
                  "Calcula indicadores; re-entrena el Random Forest cada 30 minutos y estima P(sube) a 3 velas.",
                  "Calcula VaR/CVaR 95 % (500 velas) y el filtro de volatilidad.",
                  "Si hay señal, consulta a Gemini (máx. 25 s por intento, modelos de respaldo y reglas por defecto).",
                  "Envía orden de mercado IOC; vigila SL/TP y cierra con <i>reduce_only</i> por SL, TP o tiempo.",
                  "Registra en Excel y en <i>trading_bot.log</i>, publica en Firestore y notifica por Telegram."])
    s += [p("Tecnologías", H2),
          tabla([["Componente", "Tecnología"],
                 ["Lenguaje / datos", "Python 3.10 · pandas · numpy"],
                 ["Modelo", "scikit-learn RandomForestClassifier (KNN como comparativo)"],
                 ["Exchange", "SDK gate_api · Gate.io Testnet futuros USDT"],
                 ["Agente IA", "google-genai · Gemini 3.5 Flash-Lite (respaldos 3.1 Flash-Lite y Flash)"],
                 ["Mensajería", "API HTTP de Telegram (long polling, teclado de menú)"],
                 ["Base de datos", "Firebase Firestore vía REST con JWT RS256 (cuenta de servicio)"],
                 ["App", "RiskSignal AI · Google AI Studio · Firebase Auth · onSnapshot"],
                 ["Infraestructura", "Compute Engine e2-small · Ubuntu 22.04 LTS · 20 GB · systemd"]],
                [4 * cm, 12.5 * cm])]

    # ---------------- Matemáticas ----------------
    s += [PageBreak(), p("2. Fundamento matemático", H1),
          p("2.1 Indicadores técnicos", H2),
          p("Media móvil exponencial con n ∈ {9, 21}:"),
          p("EMA<sub>t</sub> = α·P<sub>t</sub> + (1 − α)·EMA<sub>t−1</sub>,  α = 2 / (n + 1)", EQ),
          p("RSI de Wilder con n = 14, donde G y L son promedios exponenciales (α = 1/14) de ganancias y pérdidas:"),
          p("RSI<sub>t</sub> = 100 − 100 / (1 + G<sub>t</sub> / L<sub>t</sub>)", EQ),
          p("Volatilidad de corto plazo (ventana de 12 velas, como en la clase 10):"),
          p("σ<sub>12</sub> = desv.est.(P<sub>t−11</sub>, …, P<sub>t</sub>) / P<sub>t</sub> × 100", EQ),
          p("2.2 Random Forest", H2),
          p("Se modela la dirección a h = 3 velas: y<sub>t</sub> = 1[P<sub>t+3</sub> > P<sub>t</sub>]. El vector de "
            "variables x<sub>t</sub> contiene los rendimientos de 1, 2, 3, 6 y 12 velas, la brecha de medias "
            "(EMA<sub>9</sub>/EMA<sub>21</sub> − 1), el RSI, la volatilidad de 12 velas y el rango (H − L)/C. El bosque "
            "promedia B árboles entrenados sobre muestras bootstrap con selección aleatoria de variables, lo que "
            "reduce la varianza respecto a un árbol individual:"),
          p("p̂(x<sub>t</sub>) = (1/B) · Σ<sub>b=1..B</sub> p̂<sub>b</sub>(x<sub>t</sub>),   B = 200", EQ),
          p("2.3 Riesgo: VaR y CVaR históricos", H2),
          p("Sobre los últimos 500 rendimientos r de 2 minutos y nivel de confianza 95 %:"),
          p("VaR<sub>95</sub> = − q<sub>0.05</sub>(r)      CVaR<sub>95</sub> = − E[ r | r ≤ q<sub>0.05</sub>(r) ]", EQ),
          p("El CVaR (Expected Shortfall) cuantifica la pérdida promedio en el 5 % peor de los casos y se reporta "
            "como medida complementaria de riesgo de cola."),
          p("2.4 Métricas de validación", H2),
          p("Con y real, ŷ pronóstico y n observaciones de prueba:"),
          p("RMSE = √[(1/n) Σ (y − ŷ)<super>2</super>]    MAE = (1/n) Σ |y − ŷ|    "
            "MAPE = (100/n) Σ |(y − ŷ)/y|", EQ),
          p("SMAPE = (100/n) Σ 2|ŷ − y| / (|y| + |ŷ|)    Variance Score = 1 − Var(y − ŷ) / Var(y)", EQ),
          p("Para clasificación: Precision = TP/(TP + FP), Recall = TP/(TP + FN), "
            "F1 = 2·Precision·Recall/(Precision + Recall)."),
          p("2.5 Prueba de Diebold-Mariano", H2),
          p("Compara la precisión de dos pronósticos con la diferencia de pérdidas cuadráticas "
            "d<sub>t</sub> = e<sub>1,t</sub><super>2</super> − e<sub>2,t</sub><super>2</super>. Bajo H0 de igual "
            "precisión, E[d<sub>t</sub>] = 0:"),
          p("DM = d̄ / √( [γ̂<sub>0</sub> + 2 Σ<sub>k=1..h−1</sub> γ̂<sub>k</sub>] / n )", EQ),
          p("donde γ̂<sub>k</sub> es la autocovarianza de d<sub>t</sub>. Se aplica la corrección de muestra pequeña de "
            "Harvey, Leybourne y Newbold (1997), DM* = DM · √[(n + 1 − 2h + h(h − 1)/n)/n], contrastada con una t de "
            "Student con n − 1 grados de libertad. DM < 0 indica que el modelo 1 tiene menor error.")]

    # ---------------- Estrategia e hiperparámetros ----------------
    s += [PageBreak(), p("3. Estrategia de trading", H1),
          tabla([["Condición", "LONG", "SHORT"],
                 ["Tendencia", "EMA9 > EMA21", "EMA9 < EMA21"],
                 ["Momentum", "45 ≤ RSI ≤ 70", "30 ≤ RSI ≤ 55"],
                 ["Random Forest", "P(sube) ≥ 0.55", "1 − P(sube) ≥ 0.55"],
                 ["Volatilidad", "σ12 ≤ 0.60 %", "σ12 ≥ 0.03 %"],
                 ["Agente Gemini", "Confirma OPERAR", "Confirma OPERAR"]],
                [4.5 * cm, 6 * cm, 6 * cm]),
          Spacer(1, 6),
          p("4. Tabla de hiperparámetros del bot", H1),
          tabla([["Componente", "Parámetro", "Valor"],
                 ["Datos", "Temporalidad · contrato", "2 min (agregadas de 1 min) · XRP_USDT futuros"],
                 ["Indicadores", "EMA rápida / lenta · RSI", "9 / 21 · 14"],
                 ["Random Forest", "n_estimators · max_depth · min_samples_leaf", "200 · 6 · 20"],
                 ["Random Forest", "Horizonte · umbral · re-entrenamiento", "3 velas (6 min) · 0.55 · 30 min"],
                 ["Riesgo", "VaR/CVaR · ventana", "95 % histórico · 500 velas"],
                 ["Riesgo", "k_sl / k_tp por defecto · rango del agente", "2.0 / 1.0 · [0.8, 3.0] con k_tp ≤ k_sl"],
                 ["Riesgo", "SL / TP mínimos", "0.25 % / 0.15 %"],
                 ["Riesgo", "Salida por tiempo · pausa automática", "15 velas · 4 pérdidas seguidas"],
                 ["Tamaño", "Nocional · apalancamiento", "100 USDT · 1x"],
                 ["Agente", "Modelo · timeout", "gemini-3.5-flash-lite · 25 s"],
                 ["Validación", "KNN comparativo", "k = 25, variables estandarizadas"]],
                [3.2 * cm, 6.3 * cm, 7 * cm]),
          p("5. Gestión de riesgo", H1),
          p("El Stop Loss y el Take Profit se derivan del VaR, con P<sub>e</sub> el precio de entrada:"),
          p("LONG:  SL = P<sub>e</sub>(1 − k<sub>sl</sub>·VaR),  TP = P<sub>e</sub>(1 + k<sub>tp</sub>·VaR)     "
            "SHORT:  SL = P<sub>e</sub>(1 + k<sub>sl</sub>·VaR),  TP = P<sub>e</sub>(1 − k<sub>tp</sub>·VaR)", EQ),
          p("Se eligió el VaR en lugar del ATR porque mide directamente la pérdida de cola de la distribución empírica "
            "de rendimientos en la misma temporalidad en que opera el bot; ambos escalan las distancias con la "
            "volatilidad reciente. Controles adicionales:")]
    s += bullets(["Restricción k<sub>tp</sub> ≤ k<sub>sl</sub> impuesta en código aunque el agente proponga otra cosa, "
                  "y mínimos de SL/TP que cubren comisiones.",
                  "Filtro de volatilidad: sin LONG en volatilidad extrema ni SHORT en mercado sin movimiento.",
                  "Salida por tiempo a las 15 velas (30 min).",
                  "<i>Circuit breaker</i>: pausa automática tras 4 pérdidas consecutivas con aviso por Telegram.",
                  "Cierres <i>reduce_only</i>; PnL oficial de <i>list_position_close</i> (incluye comisiones).",
                  "Apalancamiento 1x y nocional fijo de 100 USDT por operación.",
                  "Secretos en <i>.env</i> y <i>firebase_key.json</i>, excluidos del repositorio."])

    # ---------------- Validación ----------------
    s += [PageBreak(), p("6. Validación del modelo", H1),
          p(f"<b>Datos:</b> API pública de Gate.io, futuros XRP_USDT, {int(meta['velas']):,} velas de 2 minutos del "
            f"{str(meta['inicio_utc'])[:16]} al {str(meta['fin_utc'])[:16]} UTC (máximo histórico de 1 minuto que "
            f"ofrece la API). <b>Partición temporal</b> 80/20 sin mezclar: {int(meta['train']):,} observaciones de "
            f"entrenamiento y {int(meta['test']):,} de prueba, con una purga de 3 velas entre ambas para evitar fuga "
            "de información por el horizonte del objetivo. Se usan exactamente las variables del bot. Para regresión "
            "se modela el rendimiento a 3 velas y se reconstruye el precio. Modelos de referencia: KNN (k = 25), "
            "caminata aleatoria (P̂<sub>t+3</sub> = P<sub>t</sub>) y momentum de una vela."),
          p("6.1 Clasificación (Buy/Sell)", H2),
          tabla([["Modelo", "Accuracy", "Precision", "Recall", "F1-Score"]] +
                [[m] + [f"{v:.4f}" for v in r] for m, r in clas.iterrows()],
                [5.5 * cm, 2.75 * cm, 2.75 * cm, 2.75 * cm, 2.75 * cm]),
          Spacer(1, 6),
          figura(RES / "matrices_confusion.png", 16.5, "Figura 2. Matrices de confusión en el conjunto de prueba."),
          p("6.2 Regresión (precio a 6 minutos)", H2),
          tabla([["Modelo", "RMSE", "MSE", "MAE", "MAPE %", "SMAPE %", "Var. Score"]] +
                [[m, f"{r['RMSE']:.6f}", f"{r['MSE']:.3e}", f"{r['MAE']:.6f}", f"{r['MAPE (%)']:.4f}",
                  f"{r['SMAPE (%)']:.4f}", f"{r['Variance Score']:.4f}"] for m, r in reg.iterrows()],
                [4.1 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2 * cm, 2 * cm, 2.1 * cm]),
          Spacer(1, 6),
          figura(RES / "comparativa_modelos.png", 15.5, "Figura 3. RMSE y MAE por modelo."),
          figura(RES / "pronostico_vs_real.png", 15.5, "Figura 4. Pronóstico a 6 minutos vs. precio real (últimas 120 "
                 "velas de prueba)."),
          p("6.3 Prueba de Diebold-Mariano", H2),
          tabla([["Modelo 1", "Modelo 2", "DM (HLN)", "p-valor", "Conclusión (α = 5 %)"]] +
                [[r["Modelo 1"], r["Modelo 2"], f"{r['DM (HLN)']:.3f}", f"{r['p-valor']:.4f}", r["Conclusión (α=5%)"]]
                 for _, r in dm.iterrows()],
                [3.5 * cm, 3.5 * cm, 1.8 * cm, 1.7 * cm, 6 * cm]),
          Spacer(1, 4),
          p("Ningún modelo supera significativamente a la caminata aleatoria; KNN es significativamente peor. Frente a "
            "KNN, el Random Forest tiene menor error, pero la diferencia no es significativa."),
          p("6.4 Simulación de hiperparámetros del Random Forest", H2),
          p("Se evaluaron 16 combinaciones de n_estimators ∈ {50, 100, 200, 400} y max_depth ∈ {4, 6, 8, sin límite}, "
            "con min_samples_leaf = 20. Para cada una se calcularon las métricas de regresión, el F1 del clasificador "
            "y la prueba DM contra la caminata aleatoria.")]
    filas = [["Árb.", "Prof.", "RMSE", "MAE", "MAPE %", "SMAPE %", "Var. Score", "F1", "Acc.", "DM", "p"]]
    for _, r in hip.iterrows():
        filas.append([int(r.n_estimators), r.max_depth, f"{r.RMSE:.6f}", f"{r.MAE:.6f}", f"{r['MAPE (%)']:.4f}",
                      f"{r['SMAPE (%)']:.4f}", f"{r['Variance Score']:.4f}", f"{r['F1-Score']:.3f}",
                      f"{r.Accuracy:.3f}", f"{r['DM vs ingenuo']:.2f}", f"{r['p-valor DM']:.3f}"])
    s += [tabla(filas, [1.35 * cm, 1.15 * cm, 1.75 * cm, 1.75 * cm, 1.5 * cm, 1.6 * cm, 1.65 * cm, 1.2 * cm, 1.6 * cm,
                        1.2 * cm, 1.35 * cm], fs=7),
          Spacer(1, 6),
          figura(RES / "hiperparametros.png", 16.5, "Figura 5. Mapas de calor de RMSE y F1 por combinación."),
          p("6.5 Interpretación", H2)]
    mejor = hip.sort_values("RMSE").iloc[0]
    s += bullets([f"El menor RMSE se obtiene con n_estimators = {int(mejor.n_estimators)} y max_depth = "
                  f"{mejor.max_depth} ({mejor.RMSE:.6f}); el número de árboles casi no cambia el error, mientras que "
                  "la profundidad lo domina.",
                  "Los árboles sin límite elevan el F1 (≈ 0.41) pero sobreajustan: su error es significativamente "
                  "mayor que el de la caminata aleatoria (p < 0.01). La configuración del bot (200, 6) queda en el punto "
                  "intermedio.",
                  "El Variance Score cercano a 0.94 refleja la persistencia del nivel de precio, no capacidad "
                  "predictiva; la referencia correcta es la caminata aleatoria, que obtiene 0.9405.",
                  "Conclusión: a 6 minutos el precio de XRP se comporta casi como una caminata aleatoria. El valor del "
                  "sistema está en combinar el filtro probabilístico con reglas de tendencia, control de volatilidad y "
                  "una gestión de riesgo estricta."])

    # ---------------- Testnet ----------------
    s += [PageBreak(), p("7. Resultados en Gate.io Testnet", H1),
          p("<i>analisis_testnet.py</i> calcula, a partir del Excel del bot, las métricas de desempeño, el "
            "<b>ROI simulado</b> sobre 100 USDT de margen por operación y la <b>matriz de confusión de las señales "
            "ejecutadas</b> (dirección operada vs. movimiento real entre entrada y salida). Corte al 19 de noviembre "
            "de 2026, primeras operaciones tras el despliegue en la VM:"),
          tabla([["Métrica", "Valor"]] + [[k, f"{int(v)}" if float(v).is_integer() and abs(v) >= 1 else f"{v:g}"]
                                          for k, v in tn.items()], [8 * cm, 4 * cm]),
          Spacer(1, 6),
          figura(RES / "testnet_resultados.png", 16, "Figura 6. Curva de capital y matriz de confusión de señales en "
                 "Testnet."),
          tabla([["#", "Entrada", "Lado", "P. entrada", "P. salida", "PnL USDT", "Salida", "P(sube)"]] +
                [[r["#"], r["Fecha entrada"], r["Posición"], r["Precio entrada"], r["Precio salida"],
                  f"{r['PnL (USDT)']:+.4f}", r["Motivo salida"], r["RF P(sube)"]] for _, r in tops.iterrows()],
                [0.8 * cm, 3.6 * cm, 1.5 * cm, 2 * cm, 2 * cm, 2 * cm, 1.8 * cm, 1.8 * cm]),
          Spacer(1, 6),
          p("La muestra todavía es pequeña para conclusiones estadísticas sobre la estrategia; el bot continúa "
            "operando 24/7 y las métricas se actualizan ejecutando el script en la VM. Las operaciones muestran el "
            "funcionamiento de punta a punta: señal, validación de Gemini con su justificación, apertura, salida por "
            "TP, SL o tiempo, y registro en Excel, Firestore y Telegram.")]

    # ---------------- Despliegue / integración ----------------
    s += [PageBreak(), p("8. Despliegue en Google Cloud", H1),
          tabla([["Elemento", "Configuración"],
                 ["Instancia", "bot-xrp-vm · e2-small (2 vCPU compartidas, 2 GB) · us-central1-f"],
                 ["Sistema / disco", "Ubuntu 22.04 LTS · 20 GB estándar"],
                 ["Ejecución 24/7", "Servicio systemd bot-xrp con Restart=always (alternativa: tmux)"],
                 ["Entorno", "venv ~/botenv instalado por setup_vm.sh"],
                 ["Secretos", ".env (Gate, Telegram, Gemini) y firebase_key.json, fuera del repositorio"],
                 ["Bitácora", "trading_bot.log con marca de tiempo por evento"],
                 ["Costo", "≈ 12 USD/mes cubierto con créditos educativos"]],
                [4 * cm, 12.5 * cm]),
          p("9. Integración con RiskSignal AI y Telegram", H1),
          p("El bot escribe con un cliente REST propio (JWT RS256 firmado con la cuenta de servicio) en la base de "
            "Firestore de la app: <i>bot_xrp/estado</i> (precio, indicadores, señal y posición), <i>bot_xrp/resumen</i> "
            "(Win Rate, PnL, profit factor, drawdown) y <i>operaciones/{n}</i>. La pestaña «Bot de trading» de "
            "RiskSignal AI se suscribe con <i>onSnapshot</i> y se actualiza en tiempo real; las reglas de seguridad "
            "permiten lectura solo a usuarios autenticados. Telegram ofrece un menú para activar o pausar el trading "
            "y consultar indicadores, predicción, parámetros del agente, operaciones, Excel, cuenta y estado."),
          p("10. Conclusiones y trabajo futuro", H1)]
    s += bullets(["Se integró de punta a punta un sistema de trading con IA: datos, modelo, riesgo, agente, ejecución, "
                  "monitoreo y despliegue continuo en la nube.",
                  "La validación rigurosa (partición temporal, purga y Diebold-Mariano) evita sobreestimar el modelo: a "
                  "6 minutos no hay ventaja predictiva significativa frente a la caminata aleatoria.",
                  "La gestión de riesgo basada en VaR, el filtro de volatilidad y el circuit breaker son el componente que "
                  "protege el capital ante la baja predictibilidad.",
                  "Trabajo futuro: horizontes de 15 a 60 minutos, variables de libro de órdenes y volumen, validación "
                  "walk-forward, y una muestra de operaciones mayor antes de considerar dinero real."])
    s += [p("Referencias", H1)]
    s += bullets(["Breiman, L. (2001). Random Forests. <i>Machine Learning</i>, 45(1), 5–32.",
                  "Diebold, F. X. y Mariano, R. S. (1995). Comparing Predictive Accuracy. <i>Journal of Business &amp; "
                  "Economic Statistics</i>, 13(3), 253–263.",
                  "Harvey, D., Leybourne, S. y Newbold, P. (1997). Testing the equality of prediction mean squared "
                  "errors. <i>International Journal of Forecasting</i>, 13(2), 281–291.",
                  "Jorion, P. (2007). <i>Value at Risk</i> (3.ª ed.). McGraw-Hill.",
                  "Wilder, J. W. (1978). <i>New Concepts in Technical Trading Systems</i>. Trend Research.",
                  f"Código, datos y resultados: {REPO}"])

    doc = SimpleDocTemplate(str(SALIDA), pagesize=letter, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=1.8 * cm, title="Reporte técnico · Bot de trading XRP/USDT con IA",
                            author="Patricio Castellanos")
    doc.build(s, onLaterPages=pie_pagina)
    print("PDF:", SALIDA)


if __name__ == "__main__":
    main()
