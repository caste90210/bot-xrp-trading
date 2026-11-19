#!/usr/bin/env bash
# Instala y arranca el bot como servicio 24/7 en la VM (Ubuntu 22.04 / Debian 12).
# Uso:  bash setup_vm.sh      (desde la carpeta donde están bot_xrp_futuros.py, .env y firebase_key.json)
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
echo ">> Instalando Python y tmux..."
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends python3.10-venv tmux
echo ">> Creando entorno virtual e instalando librerías..."
python3 -m venv "$HOME/botenv"
"$HOME/botenv/bin/pip" install --upgrade pip
"$HOME/botenv/bin/pip" install -r "$DIR/requirements.txt"
for f in .env firebase_key.json; do [ -f "$DIR/$f" ] || echo "!! Falta $f en $DIR"; done
chmod 600 "$DIR/.env" "$DIR/firebase_key.json" 2>/dev/null || true
echo ">> Creando servicio systemd bot-xrp..."
sudo tee /etc/systemd/system/bot-xrp.service >/dev/null <<UNIT
[Unit]
Description=Bot de trading XRP/USDT (Gate.io Testnet) - UDEM
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$DIR
ExecStart=$HOME/botenv/bin/python $DIR/bot_xrp_futuros.py
Restart=always
RestartSec=15
Environment=PYTHONUNBUFFERED=1
Environment=TZ=America/Monterrey

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now bot-xrp
sleep 5
sudo systemctl --no-pager status bot-xrp | head -15
echo
echo ">> Listo. Ver el log en vivo:   tail -f $DIR/trading_bot.log"
echo ">> Reiniciar / detener:         sudo systemctl restart bot-xrp   |   sudo systemctl stop bot-xrp"
