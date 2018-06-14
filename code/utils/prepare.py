from scipy import ndimage
from vggnet import Vgg19

import tensorflow as tf
import numpy as np
import data_utils
import pickle


def get_f30k_features():
    batch_size = 100

    vgg_model_path = './data/imagenet-vgg-verydeep-19.mat'

    image_dir = './flickr30k_images_resize/'

    # extract conv5_3 feature vectors
    vggnet = Vgg19(vgg_model_path)
    vggnet.build()

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        save_path = './data/f30k/f30k_%d.features.pkl'
        spilt_num = 6
        image_path = data_utils.get_f30k_name_list("./flickr30k_images_resize/")
        image_path = image_path[:30000]
        each_sum = int(30000 / spilt_num)
        for i in range(spilt_num):
            process_image = image_path[i * each_sum:(i+1) * each_sum]
            print(process_image)
            n_examples = len(process_image)
            print(n_examples)
            all_feats = np.ndarray([n_examples, 196, 512], dtype=np.float32)

            for start, end in zip(range(0, n_examples, batch_size),
                                  range(batch_size, n_examples + batch_size, batch_size)):
                image_batch_file = image_path[start:end]
                image_batch = np.array(list(map(lambda x: ndimage.imread(x, mode='RGB'), image_batch_file))).astype(
                        np.float32)
                feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})
                all_feats[start:end, :] = feats
                print("Processed %dfeatures.." % end)

            # use hickle to save huge feature vectors
            save_pickle(all_feats, save_path % i)
            print("Saved %s.." % save_path % i)


def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        print ('Loaded %s..' % path)
        return file


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print ('Saved %s..' % path)




