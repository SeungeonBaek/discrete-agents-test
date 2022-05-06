FROM python:3.8

RUN apt-get update
RUN apt-get install swig -y

RUN pip install tensorboardx tensorflow pandas gym[box2d] wandb procgen tensorflow_probability