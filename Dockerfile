ARG VARIANT="ubuntu"
FROM mcr.microsoft.com/vscode/devcontainers/cpp:${VARIANT}

RUN sudo -s

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive
RUN apt-get install g++  -y
RUN apt-get install make -y
RUN apt-get install cmake -y
RUN apt-get install git -y
# - [Install] GoogleTest and GoogleMock
RUN apt-get -y install --no-install-recommends googletest \
    && mkdir /usr/src/googletest/build/ \
    && cd /usr/src/googletest/build/ \
    && cmake -S .. -B . \
    && cmake --build . && cmake --install . \
    && rm -rf /usr/src/googletest/
# - [Install] Boost
# RUN apt-get -y install --no-install-recommends libboost-all-dev
# - [Install] OpenMPI
RUN apt-get -y install --no-install-recommends libopenmpi-dev
# - [Install] AMGCL
# RUN git clone https://github.com/ddemidov/amgcl.git /usr/src/amgcl \
#     && mkdir /usr/src/amgcl/build/ \
#     && cd /usr/src/amgcl/build/ \
#     && cmake -S .. -B . \
#     && cmake --build . && cmake --install . \
    # && rm -rf /usr/src/amgcl/

RUN exit

COPY . /workspaces/yakutat
WORKDIR /workspaces/yakutat