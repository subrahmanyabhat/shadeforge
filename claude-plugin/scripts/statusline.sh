#!/bin/bash
# ShadeForge statusline for Claude Code.
#
# Renders a live image-generation progress bar at the bottom of the terminal
# while ComfyUI is sampling, and nothing at all when it is idle.
#
# Install (~/.claude/settings.json):
#   "statusLine": {
#     "type": "command",
#     "command": "bash \"/path/to/shadeforge/claude-plugin/scripts/statusline.sh\"",
#     "refreshInterval": 2
#   }
#
# Already have a statusline? Keep it -- set SHADEFORGE_STATUSLINE_BASE to your
# existing command and this script renders yours first, then appends the bar:
#   "command": "SHADEFORGE_STATUSLINE_BASE='bash /path/to/yours.sh' bash \".../statusline.sh\""
#
# With no base command set, it renders a minimal default (directory + model).

input=$(cat)

render_default() {
  # Pull a couple of fields out of the statusline JSON without assuming jq exists.
  if command -v jq >/dev/null 2>&1; then
    printf '%s' "$input" | jq -r '
      [ (.workspace.current_dir // .cwd // "" | split("/") | last),
        (.model.display_name // "") ]
      | map(select(. != "" and . != null)) | join(" · ")' 2>/dev/null
  elif command -v python3 >/dev/null 2>&1; then
    printf '%s' "$input" | python3 -c '
import json,sys,os
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
cwd = d.get("workspace", {}).get("current_dir") or d.get("cwd") or ""
parts = [os.path.basename(cwd.rstrip("/")), (d.get("model") or {}).get("display_name") or ""]
print(" · ".join(p for p in parts if p))' 2>/dev/null
  fi
}

if [ -n "$SHADEFORGE_STATUSLINE_BASE" ]; then
  base=$(printf '%s' "$input" | eval "$SHADEFORGE_STATUSLINE_BASE" 2>/dev/null)
else
  base=$(render_default)
fi

# ShadeForge records the owning session when a generation starts. The statusline
# is global -- without this check one generation paints a bar in every terminal
# the user has open.
STATE="$HOME/.shadeforge/progress.json"
LOG=""
if [ -f "$STATE" ]; then
  owner=$(sed -n 's/.*"session"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$STATE")
  mine=$(printf '%s' "$input" | sed -n 's/.*"session_id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
  if [ -z "$owner" ] || [ "$owner" = "$mine" ]; then
    LOG=$(sed -n 's/.*"log"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$STATE")
  fi
fi

seg=""

if [ -n "$LOG" ] && [ -f "$LOG" ]; then
  # Only render while the log is actively being written, so an old run from
  # yesterday never leaves a stale bar on screen.
  now=$(date +%s)
  mtime=$(stat -f %m "$LOG" 2>/dev/null || stat -c %Y "$LOG" 2>/dev/null || echo 0)
  if [ $((now - mtime)) -lt 90 ]; then
    line=$(LC_ALL=C tail -c 4000 "$LOG" 2>/dev/null | LC_ALL=C tr '\r' '\n' \
           | LC_ALL=C grep -aE '[0-9]+/[0-9]+ \[' | tail -1)
    if [ -n "$line" ]; then
      cur=$(printf '%s' "$line" | sed -E 's|.*[^0-9]([0-9]+)/([0-9]+) \[.*|\1|')
      tot=$(printf '%s' "$line" | sed -E 's|.*[^0-9]([0-9]+)/([0-9]+) \[.*|\2|')
      eta=$(printf '%s' "$line" | sed -E 's|.*<([0-9:]+),.*|\1|')
      case "$cur$tot" in
        ''|*[!0-9]*) ;;   # not numeric -- skip
        *)
          if [ "$tot" -gt 0 ]; then
            width=16
            filled=$(( width * cur / tot ))
            pct=$(( 100 * cur / tot ))
            bar=""
            i=0
            while [ $i -lt $width ]; do
              if [ $i -lt $filled ]; then bar="${bar}█"; else bar="${bar}░"; fi
              i=$((i + 1))
            done
            sep=""
            [ -n "$base" ] && sep=$(printf ' \033[2m│\033[0m')
            seg=$(printf '%s \033[36m🎨 %s\033[0m \033[1m%d%%\033[0m \033[2m%s/%s eta %s\033[0m' \
                  "$sep" "$bar" "$pct" "$cur" "$tot" "$eta")
          fi
          ;;
      esac
    fi
  fi
fi

printf '%s%s' "$base" "$seg"
