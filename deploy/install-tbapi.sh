#!/usr/bin/env bash
# Install/refresh the telegram-bot-api service (proxychains-wrapped) and make
# sure both it and the bot are running. Idempotent — run after any change to
# deploy/telegram-bot-api.service or deploy/proxychains-tbapi.conf:
#   sudo bash /home/k/git/claude-code-telegram/deploy/install-tbapi.sh
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
cd "$(dirname "$0")"

install -m644 proxychains-tbapi.conf /etc/proxychains-tbapi.conf
cp telegram-bot-api.service /etc/systemd/system/
systemctl daemon-reload
systemctl reset-failed telegram-bot-api 2>/dev/null || true
systemctl enable --now telegram-bot-api >/dev/null 2>&1 || true
systemctl restart telegram-bot-api
echo "tbapi: $(systemctl is-active telegram-bot-api) (proxychains: $(systemctl show -p ExecStart --value telegram-bot-api | grep -c proxychains4))"

# Bring the bot back too if it crash-looped into 'failed'.
if [ "$(systemctl is-active claude-telegram-bot)" != active ]; then
  systemctl reset-failed claude-telegram-bot 2>/dev/null || true
  systemctl start claude-telegram-bot
fi
echo "bot:   $(systemctl is-active claude-telegram-bot)"
