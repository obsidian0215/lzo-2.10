#!/usr/bin/env bash
# Helper functions for creating and cleaning temporary files/dirs for lzo experiments.
# Ensures consistent prefixes and registers an EXIT trap to remove created temp dirs.


set -euo pipefail

# Array to track temp dirs created in this shell (populated when helper is used)
LZO_TMP_DIRS=()

# Install a single EXIT trap at source-time so cleanup runs in the main shell,
# not in a subshell invoked by command-substitution. This avoids races where
# a command-substitution's subshell would register a trap that fires immediately
# when the subshell exits.
if [ -z "${_LZO_TMP_CLEANUP_SET:-}" ]; then
  trap 'for d in "${LZO_TMP_DIRS[@]:-}"; do if [ -n "$d" ] && [ -d "$d" ]; then rm -rf "$d" 2>/dev/null || true; fi; done' EXIT
  _LZO_TMP_CLEANUP_SET=1
fi

# Create a temp directory with a recognizable prefix and register it for cleanup.
# Usage (preferred): lzo_mktemp_dir varname    # sets varname in the caller shell
# Usage (fallback): tmpdir=$(lzo_mktemp_dir)  # prints the path (but avoid this form)
lzo_mktemp_dir() {
  local dir
  dir=$(mktemp -d /tmp/lzo_gpu_tmp.XXXXXX) || return 1
  LZO_TMP_DIRS+=("$dir")
  if [ -n "${1:-}" ]; then
    # set caller variable by name
    eval "$1='"$dir"'"
  else
    # print to stdout (works but callers should prefer the varname form)
    printf '%s' "$dir"
  fi
}

# Create a safe temp file path (does not start with out_lzo)
# Usage: tmpfile=$(lzo_mktemp_file mylist)
lzo_mktemp_file() {
  local suffix ts file
  suffix=${1:-tmp}
  ts=$(date +%s)
  file="/tmp/lzo_${suffix}_${ts}_$$"
  : > "$file"
  printf '%s' "$file"
}

# Return a path for deletion log
lzo_deleted_log_path() {
  printf '/tmp/lzo_deleted_log_%s.txt' "$(date +%s)"
}
