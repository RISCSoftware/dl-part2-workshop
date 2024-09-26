#!/bin/bash

# # Generate a self-signed SSL certificate if it doesn't exist
# if [ ! -f /mycert.pem ]; then
#     openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
#     -keyout /mycert.key -out /mycert.pem \
#     -subj "/C=US/ST=State/L=City/O=Organization/OU=Department/CN=localhost"
# fi

TOKEN="deadbeef"

# Start Jupyter Notebook
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=/home  --IdentityProvider.token=$TOKEN &

# Start Jupyter Notebook without token
# jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --notebook-dir=/home &

# Start Jupyter Notebook with SSL
# jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --notebook-dir=/home --certfile=/mycert.pem --keyfile=/mycert.key &
