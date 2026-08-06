#!/usr/bin/env bash
set -Eeuo pipefail

SESSION="${GPU_TMUX_SESSION:-ssha800}"
TARGET="${GPU_TMUX_TARGET:-${SESSION}:1.0}"

if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "ERROR: tmux session ${SESSION} is missing." >&2
  echo "The agent is forbidden to create it or reconnect SSH; ask the user." >&2
  exit 2
fi

pane_command="$(tmux display-message -p -t "${TARGET}" '#{pane_current_command}')"
if [ "${pane_command}" != "ssh" ]; then
  echo "ERROR: ${TARGET} is not running ssh (current=${pane_command})." >&2
  echo "Stop all work and ask the user; never reconnect automatically." >&2
  exit 3
fi

echo "OK: ${TARGET} is an existing ssh pane. No network connection was created."
tmux capture-pane -t "${TARGET}" -p -S -12
