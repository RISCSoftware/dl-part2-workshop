# git stuff
cd ~/repos
git clone https://github.com/RISCSoftware/dl-part2-workshop.git
git submodule update --init --recursive
cd dl-part2-workshop

docker compose -p dl-workshop -f docker-compose.yml up -d --scale runtime-jupyter=1 --scale runtime=2
docker exec -it dl-workshop-runtime-jupyter-1 bash
docker logs dl-workshop-runtime-jupyter-1 |& grep token

# Presenter
docker exec -it dl-workshop-runtime-presenter-1 bash
jupyter notebook --ip=0.0.0.0 --port=8887 --no-browser --allow-root --notebook-dir=/repo  --IdentityProvider.token='deadbeef'

# login to registry
docker login https://index.docker.io/v1/

# image=registry.risc-software.at/risc_ds/risc_dse/dl-workshop
image=risclidse/dl-workshop
version=1.3

docker build -t $image:$version .
docker tag $image:$version $image:latest

docker push $image:$version
docker push $image:latest

docker pull $image

docker exec -it dl-workshop-runtime-1 bash
docker logs dl-workshop-runtime-1
