FROM dl-workshop-dev:latest

# Setup directory structure
ADD . /repo

# ENTRYPOINT ["tail", "-f", "/dev/null"]  # does not work with docker run -d
CMD ["tail", "-f", "/dev/null"]
