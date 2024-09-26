# git stuff
cd ~/repos
git clone https://github.com/RISCSoftware/dl-part2-workshop.git
git submodule update --init --recursive
cd dl-part2-workshop

docker compose -p dl-workshop -f docker-compose.yml up -d --scale runtime-jupyter=1 --scale runtime=2
docker exec -it dl-workshop-runtime-jupyter-1 bash
docker logs dl-workshop-runtime-jupyter-1 |& grep token

http://qftquad2.risc.jku.at:8888/tree
http://qftquad2.risc.jku.at:8888/tree/msteindl/repo/session_1
http://qftquad2.risc.jku.at:8888/notebooks/msteindl/repo/session_1/part1_preliminaries.ipynb
http://qftquad2.risc.jku.at:8888/notebooks/msteindl/repo/session_1/part2_neural_nets.ipynb

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
