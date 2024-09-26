# git stuff
cd ~/repos
git clone https://github.com/RISCSoftware/dl-part2-workshop.git
cd dl-part2-workshop

docker compose -p dl-workshop -f docker-compose.yml up -d --scale runtime-jupyter=1 --scale runtime=2

# login to registry
docker login https://index.docker.io/v1/

# image=registry.risc-software.at/risc_ds/risc_dse/dl-workshop
image=risclidse/dl-workshop
version=1.2

docker build -t $image:$version .
docker tag $image:$version $image:latest
docker build -t $image:dev-$version -f Dockerfile.dev .
docker tag $image:dev-$version $image:dev

docker push $image:$version
docker push $image:latest
docker push $image:dev-$version
docker push $image:dev

docker pull $image
docker pull $image-dev

docker exec -it dl-workshop-runtime-1 bash
docker logs dl-workshop-runtime-1
