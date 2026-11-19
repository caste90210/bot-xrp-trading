# Bot de trading XRP/USDT con IA — Gate.io Testnet + RiskSignal AI

Bot de futuros perpetuos XRP_USDT (velas de 2 min) en **Gate.io Testnet**, con EMA 9/21 + RSI 14,
**Random Forest** (probabilidad de subida), riesgo con **VaR/CVaR 95 %** y filtro de volatilidad,
agente **Gemini** que confirma señales y ajusta SL/TP, control por **Telegram** y publicación en vivo
a **Firebase Firestore** para la app **RiskSignal AI** (Google AI Studio).

## Arquitectura
```
VM Google Cloud (Compute Engine, systemd 24/7)
   bot_xrp_futuros.py ──órdenes──▶ Gate.io Testnet (futuros)
         │  ├──▶ Gemini API (validación de señal, SL/TP)
         │  ├──◀▶ Telegram (activar / pausar / estado)
         │  └──▶ Firestore (bot_xrp/estado, bot_xrp/resumen, operaciones)
         ▼
   trading_bot.log                      RiskSignal AI (Firebase Auth) lee Firestore en tiempo real
```

## Despliegue en la VM
1. VM e2-small, Ubuntu 22.04 LTS, disco 20 GB.
2. Subir: `bot_xrp_futuros.py`, `requirements.txt`, `setup_vm.sh`, `.env` (ver `.env.example`) y `firebase_key.json`.
3. `bash setup_vm.sh`  → instala todo y deja el bot como servicio `bot-xrp` (reinicio automático).
4. Log: `tail -f trading_bot.log`

## Seguridad
Las llaves van en `.env` y la clave de Firebase en `firebase_key.json`; ambos están en `.gitignore`.
