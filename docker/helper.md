sudo docker run -it \
-v /PATH_TO_IMAGE/datasets:/datasets \
image_id(eg.:0a4e7e23a9b6) python ./visual-measurement/main.py --image /datasets/bottle4.jpeg --width 2.5