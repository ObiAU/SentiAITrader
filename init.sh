#!/bin/bash
CURRENT_DIR=$(pwd)
ENV_FILE="$CURRENT_DIR/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "$ENV_FILE"
fi

if grep -q "^PYTHONPATH=" "$ENV_FILE"; then
    sed -i "s|^PYTHONPATH=.*|PYTHONPATH=$CURRENT_DIR|" "$ENV_FILE"
else
    echo "" >> "$ENV_FILE"
    echo "PYTHONPATH=$CURRENT_DIR" >> "$ENV_FILE"
fi