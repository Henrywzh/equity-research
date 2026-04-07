#!/usr/bin/env bash
# setup_oracle_vm.sh — One-time bootstrap for Oracle Cloud Free Tier ARM VM (Ubuntu 22.04)
# Run as: bash setup_oracle_vm.sh
set -euo pipefail

REPO_SSH="git@github.com:Henrywzh/equity-research.git"
REPO_DIR="$HOME/equity-research"
PROJECT_DIR="$REPO_DIR/truthsocial-monitor"
VENV_DIR="$PROJECT_DIR/.venv"
ENV_FILE="$HOME/.env"
SERVICE_NAME="truthsocial-monitor"

echo "=== [1/6] System packages ==="
sudo apt-get update -qq
sudo apt-get install -y python3.11 python3.11-venv python3-pip git sqlite3

echo "=== [2/6] Clone repo ==="
if [ ! -d "$REPO_DIR" ]; then
    git clone "$REPO_SSH" "$REPO_DIR"
else
    echo "Repo already cloned at $REPO_DIR — pulling latest."
    git -C "$REPO_DIR" pull --ff-only
fi

echo "=== [3/6] Create virtualenv and install package ==="
python3.11 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -e "$PROJECT_DIR" -q

echo "=== [4/6] Create .env file ==="
if [ ! -f "$ENV_FILE" ]; then
    cat > "$ENV_FILE" <<'EOF'
# TruthSocial credentials
TRUTHSOCIAL_USERNAME=your_truthsocial_username
TRUTHSOCIAL_PASSWORD=your_truthsocial_password

# Gmail (reuse from equity-research .config)
GMAIL_SENDER=your_gmail@gmail.com
GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
GMAIL_RECIPIENT=your_gmail@gmail.com
EOF
    echo "Created $ENV_FILE — edit it with your credentials before starting the service."
else
    echo "$ENV_FILE already exists — skipping."
fi

echo "=== [5/6] Install systemd service ==="
sudo cp "$PROJECT_DIR/deploy/$SERVICE_NAME.service" "/etc/systemd/system/$SERVICE_NAME.service"
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"

echo "=== [6/6] Done ==="
echo ""
echo "Next steps:"
echo "  1. Edit $ENV_FILE with real credentials"
echo "  2. Start the service:  sudo systemctl start $SERVICE_NAME"
echo "  3. Check status:       sudo systemctl status $SERVICE_NAME"
echo "  4. Watch logs:         journalctl -u $SERVICE_NAME -f"
echo "  5. Or tail log file:   tail -f $PROJECT_DIR/logs/daemon.log"
