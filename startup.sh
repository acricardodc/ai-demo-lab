#!/bin/bash

cd /home/site/wwwroot

python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements.txt

python -m streamlit run /home/site/wwwroot/app.py \
  --server.port 8000 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false
