FROM ubuntu:latest

# Update system
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
	&& apt-get install -y --no-install-recommends libgmp3-dev build-essential software-properties-common
RUN apt-get install -y curl

# Download and install glpk:
RUN mkdir /usr/local/glpk \
	&& curl http://ftp.gnu.org/gnu/glpk/glpk-5.0.tar.gz \
	| tar xvzC /usr/local/glpk --strip-components=1 \
	&& cd /usr/local/glpk \
	&& ./configure \
	&& make \
	&& make install

# Install python in Ubuntu
RUN apt-get install -y \
	python3.8 \
	python3-distutils \
	python3-pip \
	python3-apt
RUN add-apt-repository --enable-source --yes "ppa:marutter/rrutter4.0"
RUN add-apt-repository --enable-source --yes "ppa:c2d4u.team/c2d4u4.0+"
RUN apt install -y r-base

# Update paths:
ENV LD_LIBRARY_PATH /usr/local/lib:${LD_LIBRARY_PATH}
ENV PYTHONPATH $PYTHONPATH:.

# Install requirements:
COPY requirements ./
RUN pip3 install --upgrade pip \
	&& pip3 install -r requirements

# Final config
COPY . .
CMD ["/bin/bash"]
