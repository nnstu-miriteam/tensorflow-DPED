# python test_model.py model=iphone_orig dped_dir=dped/ test_subset=full iteration=all resolution=orig use_gpu=true

import imageio
from PIL import Image
import numpy as np
import tensorflow as tf
from models import resnet
import utils
import os
import sys
import os
import yaml
from tqdm import tqdm
import argparse


tf.compat.v1.disable_v2_behavior()

# process command arguments
#phone, dped_dir, test_subset, iteration, resolution, use_gpu = utils.process_test_model_args(sys.argv)

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='test_configs/test_config.yaml')
config_file = str(parser.parse_args().config)

with open(config_file, 'r') as file:
    cfg = yaml.safe_load(file)

use_gpu = 'true'
phone = cfg['model']
dped_dir = cfg['dped_dir']
test_subset = cfg['subset']
iteration = cfg['iteration']
resolution = 'orig'
models_folder = cfg['model_dir']
result_dir = cfg['result_dir']

if not os.path.exists(result_dir):
    os.mkdir(result_dir)

# get all available image resolutions
res_sizes = utils.get_resolutions()

# get the specified image resolution
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SIZE = utils.get_specified_res(res_sizes, phone, resolution)
# disable gpu if specified
physical_devices = tf.config.list_physical_devices('GPU')
config = tf.compat.v1.ConfigProto(device_count={'GPU': 0}) if use_gpu == "false" else None
tf.config.set_logical_device_configuration(
    physical_devices[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=10000)]
)
# create placeholders for input images
x_ = tf.compat.v1.placeholder(tf.float32, [None, IMAGE_SIZE])
x_image = tf.reshape(x_, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])

# generate enhanced image
enhanced = resnet(x_image)

with tf.compat.v1.Session(config=config) as sess:

    test_dir = dped_dir + phone.replace("_orig", "") + "/test_data/full_size_test_images/"
    test_photos = [f for f in os.listdir(test_dir) if os.path.isfile(test_dir + f)]

    if test_subset == "small":
        # use five first images only
        test_photos = test_photos[0:5]

    if phone.endswith("_orig"):

        # load pre-trained model
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, "models_orig/" + phone)

        for photo in test_photos:

            # load training image and crop it if necessary

            print("Testing original " + phone.replace("_orig", "") + " model, processing image " + photo)
            image = np.float16(np.array(Image.fromarray(imageio.imread(test_dir + photo))
                                        .resize([res_sizes[phone][1], res_sizes[phone][0]]))) / 255
            #image = np.array(Image.fromarray(imageio.imread(test_dir + photo)).resize([res_sizes[phone][1], res_sizes[phone][0]]), dtype=np.float32) / 255.0

            image_crop = utils.extract_crop(image, resolution, phone, res_sizes)
            image_crop_2d = np.reshape(image_crop, [1, IMAGE_SIZE])

            # get enhanced image

            enhanced_2d = sess.run(enhanced, feed_dict={x_: image_crop_2d})
            enhanced_image = np.reshape(enhanced_2d, [IMAGE_HEIGHT, IMAGE_WIDTH, 3]).clip(0, 1)

            before_after = np.hstack((image_crop, enhanced_image))
            photo_name = photo.rsplit(".", 1)[0]

            # save the results as .png images
            enhanced_image = (enhanced_image * 255).astype(np.uint8)
            before_after = (before_after * 255).astype(np.uint8)
            imageio.imwrite("visual_results/" + phone + "_" + photo_name + "_enhanced.png", enhanced_image)
            imageio.imwrite("visual_results/" + phone + "_" + photo_name + "_before_after.png", before_after)

    else:

        num_saved_models = int(len([f for f in os.listdir(models_folder) if f.startswith(phone + "_iteration")]) / 2)
        if iteration == "all":
            iteration = np.arange(1, num_saved_models) * 1000
        else:
            iteration = [int(iteration)]
        for i in iteration:
            # load pre-trained model
            saver = tf.compat.v1.train.Saver()
            saver.restore(sess, models_folder + "/" + phone + "_iteration_" + str(i) + ".ckpt")

            for photo in tqdm(test_photos):

                # load training image and crop it if necessary

                #print("iteration " + str(i) + ", processing image " + photo)
                image = np.float16(np.array(Image.fromarray(imageio.imread(test_dir + photo))
                                            .resize([res_sizes[phone][1], res_sizes[phone][0]]))) / 255
                #image = np.array(Image.fromarray(imageio.imread(test_dir + photo)).resize([res_sizes[phone][1], res_sizes[phone][0]]), dtype=np.float32) / 255.0

                image_crop = utils.extract_crop(image, resolution, phone, res_sizes)
                image_crop_2d = np.reshape(image_crop, [1, IMAGE_SIZE])

                # get enhanced image

                enhanced_2d = sess.run(enhanced, feed_dict={x_: image_crop_2d})
                enhanced_image = np.reshape(enhanced_2d, [IMAGE_HEIGHT, IMAGE_WIDTH, 3]).clip(0, 1)

                #before_after = np.hstack((image_crop, enhanced_image))
                photo_name = photo.rsplit(".", 1)[0]

                # save the results as .png images
                enhanced_image = (enhanced_image * 255).astype(np.uint8)
               # before_after = (before_after * 255).astype(np.uint8)
    
                imageio.imwrite(result_dir + "/" + phone + "_" + photo_name + "_iteration_" + str(i) + "_enhanced.png", enhanced_image)
                #imageio.imwrite("visual_results/" + phone + "_" + photo_name + "_iteration_" + str(i) + "_before_after.png", before_after)
