FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common=0.96.24.32.12 && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential=12.4ubuntu1 \
        cuda-command-line-tools-10-0 \
        cuda-cublas-10-0 \
        cuda-cufft-10-0 \
        cuda-curand-10-0 \
        cuda-cusolver-10-0 \
        cuda-cusparse-10-0 \
        ca-certificates=20180409 \
        curl=7.58.0-2ubuntu3.8 \
        wget=1.19.4-1ubuntu2.2 \
        libatlas-base-dev=3.10.3-5 \
        libcurl4-openssl-dev=7.58.0-2ubuntu3.8 \
        libgomp1=8.4.0-1ubuntu1~18.04 \
        libbz2-dev=1.0.6-8.1ubuntu0.2 \
        libopencv-dev=3.2.0+dfsg-4ubuntu0.1 \
        openssh-client=1:7.6p1-4ubuntu0.3 \
        openssh-server=1:7.6p1-4ubuntu0.3 \
        zlib1g-dev=1:1.2.11.dfsg-0ubuntu2 \
        protobuf-compiler=3.0.0-9.1ubuntu1 \
        libprotoc-dev=3.0.0-9.1ubuntu1 \
        libffi-dev=3.2.1-8 \
        cmake=3.10.2-1ubuntu2.18.04.1 \
        git=1:2.17.1-1ubuntu0.7 && \
    rm -rf /var/lib/apt/lists/*
    
RUN cd /usr/src && \
    curl https://www.openssl.org/source/openssl-1.0.2o.tar.gz | tar xz && \
    cd /usr/src/openssl-1.0.2o && \
    ./config --prefix=/usr/local/ssl --openssldir=/usr/local/ssl shared zlib && \
    make && make install && \
    cd /etc/ld.so.conf.d && \
    touch openssl-1.0.2o.conf && \
    echo "/usr/local/ssl/lib" > openssl-1.0.2o.conf && \
    ldconfig -v && \
    rm /usr/bin/c_rehash && rm /usr/bin/openssl

ENV PATH="${PATH}:/usr/local/ssl/bin"

RUN wget https://www.python.org/ftp/python/3.8.0/Python-3.8.0.tgz && \
        tar -xvf Python-3.8.0.tgz && cd Python-3.8.0 && \
        ./configure --enable-optimizations && \
        make && make install && \
        rm -rf ../Python-3.8.0* && \
        ln -s /usr/local/bin/pip3 /usr/bin/pip

RUN pip3 --no-cache-dir install pip==20.0.2 setuptools==42.0.2
RUN ln -s $(which python3) /usr/local/bin/python

WORKDIR /

RUN pip3 install --no-cache --upgrade --pre \
        mxnet-cu101mkl==1.6.0 \
        mlflow==1.8.0 \
        sagemaker-training==3.4.2 \
        pymysql==0.9.3 \
        pandas==1.0.3 \
        boto3==1.12.48 \
        numpy==1.18.3
