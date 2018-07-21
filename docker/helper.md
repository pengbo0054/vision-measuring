sudo docker run -it \
-v /PATH_TO_IMAGES/images:/images \
image_id(eg.:0a4e7e23a9b6) python3 ./visual-measurement/main.py --image /images/bottle4.jpeg --width 2.5