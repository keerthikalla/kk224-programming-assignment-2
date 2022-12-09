FROM datamechanics/spark:3.1-latest

ENV PYSPARK_MAJOR_PYTHON_VERSION=3

WORKDIR /opt/winequalitypred
RUN conda install numpy

COPY /kk224_test.py .
ADD ValidationDataset.csv .
Add model ./model/