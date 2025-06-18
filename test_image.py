import tensorflow as tf
import numpy as np
from pathlib import Path
from models import resnet


def process_image(img, model):
    tf.compat.v1.reset_default_graph()
    gpu = 'false'
    height, width = 3136, 4224
    IMAGE_SIZE = height * width * 3
    tf.compat.v1.disable_v2_behavior()
    config = tf.compat.v1.ConfigProto(device_count={'GPU': 0}) if gpu == 'false' else None

    x_ = tf.compat.v1.placeholder(tf.float32, [None, IMAGE_SIZE])
    x_image = tf.reshape(x_, [-1, height, width, 3])

    enhanced = resnet(x_image)

    with tf.compat.v1.Session(config=config) as sess:
        saver = tf.compat.v1.train.Saver()
        print('load saver')
        saver.restore(sess, model)
        #image = np.float16(cv2.imread(path_to_image))
        image = np.float16(img) / 255
        image_2d = np.reshape(image, [1, IMAGE_SIZE])
        print('process image')
        enhanced_2d = sess.run(enhanced, feed_dict={x_: image_2d})
        enhanced_image = np.reshape(enhanced_2d, [height, width, 3]).clip(0, 1)
        enhanced_image = (enhanced_image * 255).astype(np.uint8)
        return enhanced_image
