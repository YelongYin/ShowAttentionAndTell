import  tensorflow as tf
import utils.VGG16 as VGG
import pickle
class model():
    def __init__(self,config):
        self.batch_size=config.batch_size
        self.img_vec_file=config.img_vec_file
        self.vgg_parameter_file=config.vgg_parameter_file
        self.feature5_1_file=config.feature5_1_file
        self.feature5_2_file=config.feature5_2_file
        self.feature5_3_file=config.feature5_3_file



    def get_img_batch(self):
        image_queue=tf.train.string_input_producer(self.img_vec_file)
        reader=tf.TFRecordReader()
        _,serialized_example=reader.read(image_queue)
        features=tf.parse_single_example(
            serialized_example,
            features={
                'image':tf.VarLenFeature(tf.string)
            }
        )
        images=tf.decode_raw(features['image'],tf.float32)
        images-tf.reshape(images,[224,224,3])
        #images=tf.convert_to_tensor(images)
        img_batch=tf.train.batch(images,batch_size=self.batch_size,num_threads=32,capacity=1000)
        return img_batch
    def get_img_feature(self):
        f5_1=open(self.feature5_1_file,'w')
        f5_2=open(self.feature5_2_file,'w')
        f5_3=open(self.feature5_3_file,'w')
        img_batch=self.get_img_batch()
        vgg16=VGG(self.vgg_parameter_file)
        images=tf.placeholder('image',[None,224,224,3])
        feed_fict={images:img_batch}
        with tf.Session() as sess:
            with tf.name_scope('content_vgg'):
                vgg16.build(images)
            conv5_1_features,conv5_2_features,conv5_3_features=sess.run(
                                    vgg16.conv5_1,
                                    vgg16.conv5_2,
                                    vgg16.conv5_3,feed_fict=feed_fict)
            print(conv5_1_features.shape())
            re_conv5_1_features=tf.reshape(conv5_1_features,[-1,196,512])
            re_conv5_2_features=tf.reshape(conv5_2_features,[-1,196,512])
            re_conv5_3_features=tf.reshape(conv5_3_features,[-1,196,512])
            print(re_conv5_1_features.shape())
            pickle.dump(re_conv5_1_features,f5_1)
            pickle.dump(re_conv5_2_features,f5_2)
            pickle.dump(re_conv5_3_features,f5_3)




