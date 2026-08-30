# Local Bot API server (2 GB file uploads)

Telegram's public Bot API caps `getFile` downloads at **20 MB**. Running your own
[`telegram-bot-api`](https://github.com/tdlib/telegram-bot-api) server in `--local`
mode lifts that to **2 GB** and returns files as on-disk paths (no HTTP download).

The bot already supports this: set `TELEGRAM_BASE_URL` / `TELEGRAM_BASE_FILE_URL`
(see `src/bot/core.py`) and it switches to the local server + `local_mode`.

## 0. Credentials — DONE
api_id/api_hash are account-level (a bot has none of its own). They're already
staged at `~/.config/telegram-bot-api/env` (perms 600), reused from the realestate
`test_dot` account's `app_id`/`app_hash`. To use a different pair, overwrite that
file (or create one at <https://my.telegram.org>).

## 1. Install the server
```bash
yay -S telegram-bot-api          # AUR build; the pacman step needs sudo
```

## 2. Credentials file — DONE (see step 0)
Already at `~/.config/telegram-bot-api/env`; the unit's `EnvironmentFile` points
there. Nothing to do.

## 3. Install + start the server
```bash
sudo cp deploy/telegram-bot-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now telegram-bot-api
systemctl status telegram-bot-api          # expect: active (running), listening on 127.0.0.1:8081
```

## 4. Migrate the bot token cloud -> local
The token can't be logged in on both servers at once, so log it out of the cloud
first. (This is a brief outage — the Telegram mirror drops until step 6.)
```bash
sudo systemctl stop claude-telegram-bot
TOKEN=$(grep -E '^TELEGRAM_BOT_TOKEN=' /home/k/git/claude-code-telegram/.env | cut -d= -f2-)
curl -s -x http://localhost:1900 "https://api.telegram.org/bot$TOKEN/logOut"   # -> {"ok":true,...}
```

## 5. Point the bot at the local server
Add to `/home/k/git/claude-code-telegram/.env`:
```
TELEGRAM_BASE_URL=http://localhost:8081/bot
TELEGRAM_BASE_FILE_URL=http://localhost:8081/file/bot
```

## 6. Bring the bot back up
```bash
sudo systemctl start claude-telegram-bot
journalctl -u claude-telegram-bot -n 30 --no-pager   # expect: "Using local Bot API server"
```

Send a >20 MB video from Telegram → the bot replies `⏳ receiving …` immediately
(delivery runs off the update lock, so the bot stays responsive), then the file
lands as `video_<id>.mp4` in the bound session's cwd and the reply flips to `📎`.
The server stores files under `/home/k/.local/share/telegram-bot-api/<token>/…`;
the bot **hardlinks** them into the session cwd (instant, no extra disk) when
that store shares a filesystem with the cwd — otherwise it copies.

## Rollback (back to cloud api)
```bash
sudo systemctl stop claude-telegram-bot
curl -s "http://localhost:8081/bot$TOKEN/logOut"     # log out of the local server
sudo systemctl disable --now telegram-bot-api
# remove the two TELEGRAM_BASE_* lines from .env
sudo systemctl start claude-telegram-bot
```

## Notes
- Egress: `--proxy` only covers *webhook* requests — tdlib's MTProto to the DCs
  ignores it and env proxies. The unit therefore wraps the binary in
  `proxychains4 -f /etc/proxychains-tbapi.conf` (socks5://127.0.0.1:1900), which
  forces every connect() through the explicit proxy. No TUN needed.
- Creds are read from the env file (`TELEGRAM_API_ID`/`TELEGRAM_API_HASH`), not
  passed as flags, so they don't show up in `ps`.
- The 20 MB cap removal + video handler (this same change set) work regardless of
  local mode; local mode is only needed for files **>20 MB**.
