import cv2
import argparse
from config import Config
from utils import mark_countor, visual


def main(args):

    (image, cnts, pixelsPerMetric) = mark_countor(args['image'])

    for cnt in cnts:
        # ignore the contour if the area is small(noise)
        if cv2.contourArea(cnt) < Config.AREA_THRESHOLD:
            continue

        visual(image, cnt, pixelsPerMetric, args['width'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, required=True, help="path to the demo image")
    parser.add_argument("-w", "--width", type=float, required=True, help="width of the left-most object in the image (in centimeter)")
    args = vars(parser.parse_args())
    main(args)
