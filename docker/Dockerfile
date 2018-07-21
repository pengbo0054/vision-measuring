FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends git python3-pip wget python-opencv
RUN pip3 install --upgrade pip && pip install opencv-python scikit-learn matplotlib scipy
RUN pip install --upgrade imutils
RUN git clone --branch master https://github.com/pengbo0054/visual-measurement.git
VOLUME ["/images"]
