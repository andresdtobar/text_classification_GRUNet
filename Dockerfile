FROM ubuntu
RUN apt update
RUN apt install -y python3.9
RUN apt install -y python3-pip
RUN apt install -y python-dev
RUN apt install -y libhunspell-dev
RUN apt install -y unixodbc-dev

WORKDIR /Modelo
COPY . /Modelo/

RUN pip3 install --user pyodbc
RUN pip install -r requirements.txt
RUN python3 -m spacy download es_core_news_md

RUN apt install -y curl
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN curl https://packages.microsoft.com/config/ubuntu/20.04/prod.list > /etc/apt/sources.list.d/mssql-release.list
RUN apt-get update
RUN ACCEPT_EULA=Y apt-get install -y msodbcsql17

#VOLUME ["/Webscraping/EVIDENCIAS"]

CMD ["python3","main.py", "Train"]