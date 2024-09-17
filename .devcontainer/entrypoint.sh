#!/bin/bash

# Create user directories
mkdir -p /home/msteindl /home/jdoe

# Copy the repository to all user directories
find /home -mindepth 1 -maxdepth 1 -type d | while read SUBDIR; do
  cp -r "/repo" "$SUBDIR"
done

# Start Jupyter Notebook
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --notebook-dir=/home &

# Keep the container running
tail -f /dev/null
