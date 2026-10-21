#!/bin/zsh
# System-wide used memory in GB (wired + active + compressed + file-backed), from vm_stat. Works for engines whose
# weights are mmapped into Metal buffers, which a per-process footprint does not count.
vm_stat | awk '/page size of/ {ps=$8} /Pages wired down/ {w=$4} /Pages active/ {a=$3} /occupied by compressor/ {c=$5} /File-backed pages/ {f=$3}
  END {printf "%.1f wired=%.1f active=%.1f compressed=%.1f filebacked=%.1f\n", (w+a+c)*ps/1e9, w*ps/1e9, a*ps/1e9, c*ps/1e9, f*ps/1e9}'
