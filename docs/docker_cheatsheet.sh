cd ~/repos
git clone git@github.com:RISCSoftware/dl-part2-workshop.git
cd dl-part2-workshop

# docker pull huggingface/transformers-pytorch-gpu:latest

docker build -t dl-workshop .; docker build -t dl-workshop-dev -f Dockerfile.dev .
docker compose -p dl-workshop -f docker-compose.yml up -d --scale runtime-jupyter=1 --scale runtime=2
docker image prune -f

# login to registry
docker login https://index.docker.io/v1/

registry=registry.risc-software.at/risc_ds/risc_dse
# registry=risclidse/dl-workshop-part2
version=1.1

docker build -t $registry/dl-workshop:$version .; docker tag $registry/dl-workshop:$version $registry/dl-workshop:latest
docker build -t $registry/dl-workshop-dev:$version -f Dockerfile.dev .; docker tag $registry/dl-workshop-dev:$version $registry/dl-workshop-dev:latest

docker push $registry/dl-workshop:$version
docker push $registry/dl-workshop:latest
docker push $registry/dl-workshop-dev:$version
docker push $registry/dl-workshop-dev:latest

docker pull $registry/dl-workshop
docker pull $registry/dl-workshop-dev

docker exec -it dl-workshop-dev bash
docker logs dl-workshop-dev
