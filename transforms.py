import argparse
from PIL import Image
import math
from joblib import Parallel, delayed
from threading import Thread, Lock

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

argparser = argparse.ArgumentParser(description='Various transforms of images')
argparser.add_argument('operation', help='Operation to execute on image')
argparser.add_argument('image', help='Input image to do operation on')
args = argparser.parse_args()

imagepath = args.image
image = Image.open(imagepath)

if (not image):
    print('Could not open image "{}"'.format(imagepath))
    exit(0)

log_normal = 255 / math.log(255)
sqrt_normal = 255 / math.sqrt(255)
pow_normal = 255 / math.pow(255, 2)
invlog_normal = 255 / math.pow(10, 255)
gamma = 1 / 2.5
gamma_scale = 255 / math.pow(255, gamma)
bitplane = 1 << (8 - 1)

def process_point(i):
    x = i % image.width
    y = int(i / image.width)

    # Grayscales only return an int, not a tuple
    orig_pixel = image.getpixel((x, y))
    if not isinstance(orig_pixel, tuple):
        orig_pixel = (orig_pixel,)
    new_pixel = []

    for channel in orig_pixel:
        #new_pixel.append(int(math.log(1 + channel) * log_normal))
        #new_pixel.append(int(math.sqrt(channel) * sqrt_normal))
        #new_pixel.append(channel)
        #new_pixel.append(int(math.pow(channel, 2) * pow_normal))
        #new_pixel.append(int(math.pow(10, channel) * invlog_normal))
        #new_pixel.append(255 - channel)
        #new_pixel.append(int(math.pow(channel, gamma) * gamma_scale))
        #if channel > 100 and channel < 200:
        #    new_pixel.append(101)
        #else:
        #    new_pixel.append(channel)
        if channel & bitplane:
            new_pixel.append(channel)
        else:
            new_pixel.append(0)

    # Grayscales only take an int, not a tuple
    if len(new_pixel) == 1:
        new_pixel = new_pixel[0]
    else:
        new_pixel = tuple(new_pixel)
    image.putpixel((x, y), new_pixel)

#Parallel(n_jobs=8)(delayed(process_point)(i) for i in range(0, image.height * image.width))
for i in range(0, image.height * image.width):
    process_point(i)

image.show()

print("Finished")
