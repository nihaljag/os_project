FROM python:3

ADD trained_faces2.0.yml / 
ADD haar_face.xml /
ADD 1.jpg /
RUN apt-get update && apt-get install -y libgl1
RUN pip3 install opencv-python-headless opencv-contrib-python
ADD recognizer.py / 
CMD [ "python3", "./recognizer.py" ]