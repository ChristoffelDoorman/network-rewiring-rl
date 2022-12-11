FROM ubuntu:18.04
SHELL ["/bin/bash", "-c"]

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

RUN apt-get update && apt-get install -y netcat iproute2 dnsutils vim git build-essential cmake sqlite3 libsqlite3-dev zip unzip \
    libboost-dev libboost-system-dev libboost-thread-dev libboost-serialization-dev \
    libboost-python-dev libboost-regex-dev libboost-numpy-dev

RUN apt-get update && apt-get install -y graphviz libgraphviz-dev graphviz-dev pkg-config

WORKDIR /relnet
ADD ./environment.yml /relnet/environment.yml
RUN conda env update --file environment.yml

RUN cd /usr/lib \
    && git clone https://github.com/VictorDarvariu/pytorch_structure2vec/ \
    && cd pytorch_structure2vec \
    && cd s2v_lib \
    && make -j4

RUN chmod -R 777 /usr/lib/pytorch_structure2vec

ENV PYTHONPATH="/usr/lib/pytorch_structure2vec/s2v_lib:${PYTHONPATH}"

RUN source activate ucfadar-relnet \
    && pip install torch==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install cffi==1.11.2

###
ARG RN_GNAME
ARG RN_GID

RUN if ! [[ $(getent group ${RN_GNAME}) ]]; \
    then groupadd -g ${RN_GID} ${RN_GNAME}; \
    fi

RUN if ! [[ $(getent passwd relnetmanager) ]]; \
    then useradd -r -u 12954 --create-home --home-dir /home/relnetmanager -g ${RN_GNAME} relnetmanager; \
    fi

RUN chgrp -R ${RN_GNAME} /relnet
RUN chmod -R g+rwx /relnet
###

# See https://askubuntu.com/questions/909277/avoiding-user-interaction-with-tzdata-when-installing-certbot-in-a-docker-contai
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y dvipng texlive-latex-extra texlive-fonts-recommended texlive-fonts-extra

RUN source activate ucfadar-relnet; python -m ipykernel install --user --name relnet --display-name "python-relnet"

RUN source activate ucfadar-relnet \
    && pip install geopy==2.0.0 \
    && pip install pyproj==2.6.1

RUN apt-get update && apt-get install -y libgeos-dev
RUN source activate ucfadar-relnet \
    && pip install git+https://github.com/matplotlib/basemap.git@3076ec9470cf7dba523bc94ebe5ae9a990e34d08 \
    && pip install pillow==7.2.0

USER relnetmanager
ENV PATH /opt/conda/envs/ucfadar-relnet/bin:$PATH
ENV RELNET_DEVICE_PLACEMENT CPU
# ports for Tensorboard, jupyter
EXPOSE 6006 8888
CMD ["bash", "/relnet/docker/relnet/start_manager.sh"]