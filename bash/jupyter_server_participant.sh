#!/bin/bash

# TOKEN=""
TOKEN="deadbeef"

# Start Jupyter Notebook
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=/home  --IdentityProvider.token=$TOKEN &
