import pickle
import  os
import  tensorflow as tf

import utils.VGG16 as VGG16
from config.Enconfig import Enconfig
import utils.data_utils as data_utils


class model():
    def __init__(self,):

        self.batch_size = Enconfig.batch_size
        self.img_vec_file=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))+'/data/'+Enconfig.img_vec_file
        self.vgg_parameter_file=os.path.dirname(os.path.dirname(__file__))+'/config/'+Enconfig.vgg_parameter_file
        self.feature5_1_file=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))+'/data/'+Enconfig.feature5_1_file
        self.feature5_2_file=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))+'/data/'+Enconfig.feature5_2_file
        self.feature5_3_file=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))+'/data/'+Enconfig.feature5_3_file
        self.img_path=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))+'/flickr30k_images/'
        self.file_path=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))+'/results_20130124.token'
        '''
        if not os.path.exists(self.feature5_1_file):
            os.makedirs(self.feature5_1_file)
        if not os.path.exists(self.feature5_2_file):
            os.makedirs(self.feature5_2_file)
        if not os.path.exists(self.feature5_3_file):
            os.makedirs(self.feature5_3_file)
         '''

    def get_img_batch(self):
        if not os.path.exists(self.img_vec_file):
            print("-------output the vector of the image to TFrecored file")
            data_utils.get_image_vec(self.file_path,self.img_path,self.img_vec_file)
        print("the vector of image has exits")

        image_queue=tf.train.string_input_producer([self.img_vec_file])
        reader=tf.TFRecordReader()
        _,serialized_example=reader.read(image_queue)
        features=tf.parse_single_example(
            serialized_example,
            features={
                'image':tf.FixedLenFeature([],tf.string)
            }
        )
        images = tf.image.decode_jpeg(features['image'])
        images = tf.image.convert_image_dtype(images, dtype=tf.float32)
        images = tf.reshape(images,[224,224,3])
        #images=tf.convert_to_tensor(images)
        img_batch=tf.train.batch([images],batch_size=self.batch_size,num_threads=32,capacity=1000)
        print("get image batch complete!")
        return img_batch

    def get_img_feature(self,):
        if not os.path.exists(self.feature5_1_file or self.feature5_2_file or self.feature5_3_file):
            img_batch = self.get_img_batch()
            f5_1 = open(self.feature5_1_file, 'w')
            f5_2 = open(self.feature5_2_file, 'w')
            f5_3 = open(self.feature5_3_file, 'w')

            vgg16 = VGG16.Vgg16(self.vgg_parameter_file)
            in_img = tf.placeholder('float32',[None,224,224,3])
            feed_dict = {in_img: img_batch}
            print("start to get image features")
            with tf.Session() as sess:
                with tf.name_scope('content_vgg'):
                    vgg16.build(in_img)
                conv5_1_features = sess.run(vgg16.conv5_1, feed_dict=feed_dict)
                conv5_2_features = sess.run(vgg16.conv5_2, feed_dict=feed_dict)
                conv5_3_features = sess.run(vgg16.conv5_3, feed_dict=feed_dict)
                print(conv5_1_features.shape())
                re_conv5_1_features=tf.reshape(conv5_1_features,[-1,196,512])
                re_conv5_2_features=tf.reshape(conv5_2_features,[-1,196,512])
                re_conv5_3_features=tf.reshape(conv5_3_features,[-1,196,512])
                print(re_conv5_1_features.shape())
                pickle.dump(re_conv5_1_features,f5_1)
                pickle.dump(re_conv5_2_features,f5_2)
                pickle.dump(re_conv5_3_features,f5_3)
            print("image features has exits!")


with tf.Session() as sess:
    m=model()
    #m.get_img_batch()
    m.get_img_feature()
   # all_img_file=m.get_img_vec_file()
    #sess.run(all_img_file)



