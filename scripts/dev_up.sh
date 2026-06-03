#!/usr/bin/env bash
# Wave 11: one-shot local stack bring-up.
#
# Brings up the docker-compose stack, waits for every service's /health
# (or container health probe) to come back green, then seeds a small
# corpus and fires a sample query against the gateway. Intended as the
# fastest possible path from a fresh clone to "answers come back".
#
# Usage:
#   scripts/dev_up.sh               # full bring-up + seed + sample query
#   scripts/dev_up.sh --no-seed     # just stand the services up
#   scripts/dev_up.sh --no-query    # bring up + seed, skip the sample query
#   scripts/dev_up.sh --profile ui  # also start the Streamlit UI on :8501
#   SKIP_BUILD=1 scripts/dev_up.sh  # `docker compose up -d` without --build
#
# Linux-only by project convention; uses bash builtins + curl + docker.
set -Eeuo pipefail

cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------

DO_SEED=1
DO_QUERY=1
EXTRA_COMPOSE_ARGS=()
COMPOSE_PROFILES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-seed)   DO_SEED=0; shift ;;
        --no-query)  DO_QUERY=0; shift ;;
        --profile)
            shift
            [[ $# -gt 0 ]] || { echo "--profile requires a value" >&2; exit 2; }
            COMPOSE_PROFILES+=("$1")
            shift
            ;;
        -h|--help)
            # Print the leading docstring block (everything from the first
            # ``# `` line up to the first blank line after the shebang).
            awk 'NR==1 { next } /^$/ { exit } /^# / { sub(/^# /, ""); print; next } /^#$/ { print ""; next }' "$0"
            exit 0
            ;;
        *)
            EXTRA_COMPOSE_ARGS+=("$1")
            shift
            ;;
    esac
done

# Defaults overridable from the environment so the script doubles as a
# template for ad-hoc invocations (handy for switching ports/models).
GATEWAY_URL="${GATEWAY_URL:-http://127.0.0.1:8000}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-300}"
WAIT_SLEEP_SECONDS="${WAIT_SLEEP_SECONDS:-3}"

log() { printf '[dev_up] %s\n' "$*"; }

require() {
    command -v "$1" >/dev/null 2>&1 || { echo "missing dependency: $1" >&2; exit 1; }
}

require docker
require curl

# ---------------------------------------------------------------------------
# Compose bring-up
# ---------------------------------------------------------------------------

profile_args=()
for prof in "${COMPOSE_PROFILES[@]}"; do
    profile_args+=("--profile" "$prof")
done

build_flag="--build"
if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
    build_flag=""
fi

log "starting docker compose (profiles: ${COMPOSE_PROFILES[*]:-none})"
# shellcheck disable=SC2086  # we want word-splitting on $build_flag
docker compose "${profile_args[@]}" up $build_flag -d "${EXTRA_COMPOSE_ARGS[@]}"

# ---------------------------------------------------------------------------
# Wait for every service's /health to return 200
# ---------------------------------------------------------------------------

wait_for_http() {
    local label=$1 url=$2 deadline=$(( SECONDS + WAIT_TIMEOUT_SECONDS ))
    log "waiting for ${label} at ${url}"
    while (( SECONDS < deadline )); do
        if curl --fail --silent --show-error --max-time 5 "${url}" >/dev/null 2>&1; then
            log "${label} ready"
            return 0
        fi
        sleep "${WAIT_SLEEP_SECONDS}"
    done
    echo "[dev_up] ${label} never reached ready state at ${url}" >&2
    docker compose logs --tail 80 "${label}" >&2 || true
    return 1
}

wait_for_http ingest  "http://127.0.0.1:8001/health"
wait_for_http rag     "http://127.0.0.1:8002/health"
wait_for_http gateway "${GATEWAY_URL}/health"

# Ollama doesn't expose /health under the same path; the compose file
# uses a CLI-based healthcheck instead, so we wait for the model puller
# container to exit successfully.
log "waiting for ollama-model puller to finish"
puller_id=$(docker compose ps -q ollama-model || true)
if [[ -n "${puller_id}" ]]; then
    # `docker wait` returns the puller's exit code; treat anything > 0
    # as a hard failure so we don't silently seed before models exist.
    puller_status=$(docker wait "${puller_id}" || echo 1)
    if [[ "${puller_status}" != "0" ]]; then
        echo "[dev_up] ollama-model exited with status ${puller_status}" >&2
        docker compose logs ollama-model >&2 || true
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Seed + sample query (delegated to scripts/seed.py so the Python side
# stays unit-testable in isolation)
# ---------------------------------------------------------------------------

PY_BIN="${PY_BIN:-../sorakaienv/bin/python}"
if [[ ! -x "${PY_BIN}" ]]; then
    PY_BIN="$(command -v python3.12 || command -v python3 || command -v python)"
fi

seed_args=("scripts/seed.py" "--gateway-url" "${GATEWAY_URL}")
if [[ "${DO_QUERY}" == "0" ]]; then
    seed_args+=("--no-query")
fi
if [[ -n "${GATEWAY_API_KEY:-}" ]]; then
    seed_args+=("--api-key" "${GATEWAY_API_KEY}")
fi

if [[ "${DO_SEED}" == "1" ]]; then
    log "seeding sample corpus via ${PY_BIN} scripts/seed.py"
    "${PY_BIN}" "${seed_args[@]}"
else
    log "skipping seed (--no-seed)"
fi

log "stack ready - gateway: ${GATEWAY_URL}/docs"
