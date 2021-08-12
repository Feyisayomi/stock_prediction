FROM ubuntu:20.10

RUN apt update && \
    apt install -y python3 python3-pip

# RUN sed -i -re 's/([a-z]{2}\.)?archive.ubuntu.com|security.ubuntu.com/old-releases.ubuntu.com/g' /etc/apt/sources.list && \
#     apt update && \
#     apt remove gcc gcc-8 && \
#     apt install -y gcc-9 && \
#     rm -f /usr/bin/gcc && \
#     ln -s /usr/bin/gcc-9 /usr/bin/gcc && \
#     apt install -y python3 python3-pip

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .

ENTRYPOINT ["python3", "app.py"]