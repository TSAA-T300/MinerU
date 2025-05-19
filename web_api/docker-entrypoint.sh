#!/bin/bash

# Command to run FastAPI using Uvicorn, pointing to app.py and binding to 0.0.0.0:8000
set -e
source /opt/mineru_venv/bin/activate
exec uvicorn app:app --host 0.0.0.0 --port 8000