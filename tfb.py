import os
import scipy.misc
import tensorflow as tf

def save_images_from_event(fn, tag, output_dir='./'):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    image_str = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str)

    sess = tf.InteractiveSession()
    with sess.as_default():
        count = 0
        for e in tf.train.summary_iterator(fn):
            for v in e.summary.value:
                if v.tag.find(tag) !=-1:
                    im = im_tf.eval({image_str: v.image.encoded_image_string})
                    output_fn = os.path.realpath('{}/image_{:05d}.png'.format(output_dir, count))
                    print(type(output_fn))
                    print("Saving '{}'".format(output_fn))
                    scipy.misc.imsave(output_fn, im)
                    count += 1  


save_images_from_event('logs/joint_50_/None_laser_0_immortal_1_sharelatent_1_noise_0/events.out.tfevents.1537941795.asr02.local', "map_4_test_0_rewards", "plot/joint_50_/")