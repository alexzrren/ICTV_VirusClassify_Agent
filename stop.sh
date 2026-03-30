#!/usr/bin/env bash
# Stop the ICTV Classifier web server.
# Usage: bash stop.sh [port]

PORT=${1:-18231}

PIDS=$(lsof -ti:"$PORT" 2>/dev/null)
if [ -z "$PIDS" ]; then
    echo "No process found on port $PORT."
    exit 0
fi

echo "Stopping ICTV Classifier on port $PORT (PIDs: $PIDS) ..."
echo "$PIDS" | xargs kill 2>/dev/null
sleep 1

# Force kill if still running
REMAIN=$(lsof -ti:"$PORT" 2>/dev/null)
if [ -n "$REMAIN" ]; then
    echo "Force killing remaining processes ..."
    echo "$REMAIN" | xargs kill -9 2>/dev/null
fi

echo "Stopped."
