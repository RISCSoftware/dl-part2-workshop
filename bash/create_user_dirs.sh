#!/bin/bash

# Read the list of workshop users
mapfile -t users < /repo/users.list

# Create a home directory for each user
for user in "${users[@]}"; do
  mkdir -p "/home/$user"
  cp -r "/repo" "/home/$user"
done
