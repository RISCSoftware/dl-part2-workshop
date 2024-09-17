#!/bin/bash

# Read the list of workshop users
mapfile -t users < /repo/.devcontainer/users.list

# Create a home directory for each user
for user in "${users[@]}"; do
  mkdir -p "/home/$user"
  cp -r "/repo" "/home/$user"
done

# Start Jupyter Notebook
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --notebook-dir=/home &

# Keep the container running
tail -f /dev/null
