FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common=0.96.24.32.12 && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
    	python3.8=3.8.2-1+bionic1 \
	    python3-pip=9.0.1-2.3~ubuntu1.18.04.1 \
        build-essential=12.4ubuntu1 \
        python3-dev=3.6.7-1~18.04 && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf python3.8 /usr/bin/python
RUN pip3 --no-cache-dir install pip==20.0.2 setuptools==44.0.0

WORKDIR /

RUN pip3 install --no-cache --upgrade \
        jupyter==1.0.0 \
        mlflow==1.8.0 \
        boto3==1.12.48 \
        pandas==1.0.3 \
        scikit-learn==0.22.2.post1 \
        sagemaker==1.55.4 \
        pymysql==0.9.3 \
        mxnet==1.6.0

WORKDIR /notebooks
CMD jupyter notebook \
    --ip=0.0.0.0 \
    --allow-root \
    --no-browser \
    --NotebookApp.token='' \
    --NotebookApp.password=''
