#!/bin/bash

# Start Jupyter Notebook
jupyter notebook --ip=0.0.0.0 --port=8887 --no-browser --allow-root --notebook-dir=/home  --IdentityProvider.token='' &
