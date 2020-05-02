FROM python:3.8.0

RUN pip install \
    mlflow==1.8.0 \
    pymysql==0.9.3 \
    boto3==1.12.48

CMD mlflow ui \
    --host 0.0.0.0 \
    --backend-store-uri mysql+pymysql://$USER:$PASSWORD@$HOST:3306/mlflow
