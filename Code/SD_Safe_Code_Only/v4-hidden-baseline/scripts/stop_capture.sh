#!/usr/bin/env bash
# kill only the capture workers, never this script's own command line
pattern='sdsafe_v4[.]capture'
pkill -TERM -f "$pattern" && echo "sent TERM" || echo "no capture process"
sleep 5
pkill -KILL -f "$pattern" 2>/dev/null || true
pgrep -af "$pattern" || echo "capture stopped"
