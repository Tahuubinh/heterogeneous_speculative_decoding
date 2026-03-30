#!/bin/bash

SCRIPT_USER="${SUDO_USER:-$USER}"
VENV_PATH="/bigtemp/${SCRIPT_USER}/nlp_venv"

mkdir -p "$VENV_PATH"
python -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt