#!/usr/bin/env bash
# Post-merge hook for agm-cli.
# Called by the agm merge pipeline after a task branch is merged to main.
# Receives the merge SHA via AGM_MERGE_SHA, runs in the project directory.
#
# Stamps _build_meta.py with the merge SHA and rebuilds the global
# `agm` binary so downstream consumers (agm-web, agents) pick up
# the new code immediately.

set -euo pipefail

MERGE_SHA="${AGM_MERGE_SHA:?AGM_MERGE_SHA is required}"
SHORT_SHA="${MERGE_SHA:0:8}"
META_FILE="src/agm/_build_meta.py"

echo "Stamping build metadata: ${SHORT_SHA}"
printf 'BUILD_SHA = "%s"\n' "${SHORT_SHA}" > "${META_FILE}"

echo "Rebuilding global agm binary"
uv tool install --from . agm --force --reinstall

echo "Post-merge complete (${SHORT_SHA})"
