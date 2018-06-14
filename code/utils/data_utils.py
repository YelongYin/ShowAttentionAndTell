from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import gfile
from config.Enconfig import Enconfig

import re
import json
import tensorflow as tf
import numpy as np
import pickle
import os

IMAGE_PATH = "/home/lemin/1TBdisk/PycharmProjects/ShowAttentionAndTell/flickr30k_images_resize/"
IMAGE_CAPTIONS_FILE = "/home/lemin/1TBdisk/PycharmProjects/ShowAttentionAndTell/results_20130124.token"

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile("\d")


def get_captions(file_path, dataset="coco"):
    """
        file_path:path of captions file
        dataset: dataset must in ["coco","flickr30k"]
        return :lists of captions
    """
    captions = []
    dataset = dataset.lower()
    if dataset == "coco":
        with open(file_path) as f:
            caption_data = json.load(f)
        for annotation in caption_data["annotations"]:
            captions.append(annotation["caption"])
        return captions
    if dataset == "flickr30k":
        with open(file_path) as f:
            for line in f.readlines():
                caption = line.split()[1:]
                caption = " ".join(caption)
                captions.append(caption)
            return captions


def get_img_caption(file_path, dataset="coco"):
    """
    This function is use to get map of image_id:caption,like {185832:  'a guy playing wii while his friends watch him'}
    which mean the 185832 image describe  a guy playing wii while his friends watch him;
    Args:
        file_path: location of your caption file just like where is your captions_train2014.json
        dataset:
    Returns:
         A map of {iamge_id: caption}
    """
    img_caption = {}
    if dataset == "coco":
        with open(file_path) as f:
            caption_data = json.load(f)
        annotations = caption_data["annotations"]
        for annotation in annotations:
            img_caption[annotation['image_id']] = annotation['caption']
    elif dataset == "flickr30k":
        with open(file_path) as f:
            for line in f.readlines():
                name_caption = line.split()
                image_name = name_caption[0][:-2]
                caption = " ".join(name_caption[1:])
                img_caption[image_name] = caption
    return img_caption


def basic_tokenizer(sentence):
    """
        Very basic tokenizer: split the sentence into a list of tokens.
    """
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]


def create_vocabulary(vocabulary_path, captions_list, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
    """Create vocabulary file (if it does not exist yet) from disc_data file.

    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.

    Args:
      vocabulary_path: path where the vocabulary will be created.
      captions_list: list of captions that will be used to create vocabulary.
      max_vocabulary_size: limit on the size of the created vocabulary.
      tokenizer: a function to use to tokenize each disc_data sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    Return:
        vocab_list
    """
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s " % vocabulary_path)
        vocab = {}
        counter = 0
        for caption in captions_list:
            counter += 1
            caption = caption.lower()
            if counter % 10000 == 0:
                print("processing line %d" % counter)
            caption = tf.compat.as_bytes(caption)
            tokens = tokenizer(caption) if tokenizer else basic_tokenizer(caption)
            for w in tokens:
                word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
        # put the special word at the start, and sort words by their frequency
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + b"\n")
        return vocab_list


def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.

    We assume the vocabulary is stored one-item-per-line, so a file:
      dog
      cat
    will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
    also return the reversed-vocabulary ["dog", "cat"].

    Args:
      vocabulary_path: path to the file containing the vocabulary.

    Returns:
      a pair: the vocabulary (a dictionary mapping string to integers), and
      the reversed vocabulary (a list, which reverses the vocabulary mapping).

    Raises:
      ValueError: if the provided vocabulary_path does not exist.
    """
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x.decode(), y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
    """Convert a string to list of integers representing token-ids.

    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

    Args:
      sentence: the sentence in bytes format to convert to token-ids.
      vocabulary: a dictionary mapping tokens to integers.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.

    Returns:
      a list of integers, the token-ids for the sentence.
    """

    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(_DIGIT_RE.sub("0", w), UNK_ID) for w in words]


def load_glove(glove_path_directory, dim=100):
    """
    Args:
        glove_path: path of glove
        dim:dimension of embedding matrix
    Return:
        glove embedding matrix
    """
    word2vec = {}
    print("==> loading glove")
    with open(glove_path_directory + "/glove.6B.%s.txt" % str(dim)) as f:
        for line in f:
            l = line.split()
            word2vec[l[0]] = list(map(float, l[1:]))
    print("==> glove is loaded")
    return word2vec


def get_image_vec(file_path, img_path, img_vec_file, dataset="flickr30k"):
    """
    from dictory obtain image path then convert to image vector,and writer to TFrecored file
    :param file_path:
    :param img_vec_file:
    :return:
    """
    img_caption = get_img_caption(file_path, dataset=dataset)
    writer = tf.python_io.TFRecordWriter(img_vec_file)
    counter = 0
    for key in img_caption:
        counter += 1
        if counter % 25 == 0:
            print("Processing %d images." % counter)
        if dataset == "coco":
            image = tf.gfile.FastGFile(img_path+'COCO_train2014_'+str(key).zfill(12)+'.jpg', 'rb').read()
        elif dataset == "flickr30k":
            image = tf.gfile.FastGFile(img_path + str(key), 'rb').read()
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
        image = tf.image.per_image_standardization(image)
        image = image.eval().tostring()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
                }
            )
        )

        writer.write(example.SerializeToString())
    print('Done!')
    writer.close()


def create_vector(word, word2vec, embed_size, silent=True):
    """
        if the word is missing from Glove, create some fake vector and store in glove!
    Args:
        word: word like "dog","cat" ,etc
        word2vec: glove or word2vec matrix
        embed_size
        silent: whether print information
    """
    vector = np.random.uniform(0.0, 1.0, (embed_size,))
    word2vec[word] = vector
    if not silent:
        print("utils.py::create_vector => %s is missing" % word)
    return vector


def create_embedding(word2vec, vocab_list, embed_size):
    """
        create embedding matrix if use glove or word2vec
    Args:
        word2vec: glove or word2vec matrix
        vocab_list: vocab list
        embed_size
    Return:
        embedding_matrix
    """
    embedding = np.zeros((len(vocab_list), embed_size))
    for i in range(len(vocab_list)):
        word = vocab_list[i]
        embedding[i] = word2vec[word]
    return embedding


def get_f30k_name_list(image_path, just_file_name=False):
    image_list = []
    image_path_dir = image_path
    for l1, l2, l3 in os.walk(image_path):
        for p in l3:
            if just_file_name:
                image_list.append(p)
            else:
                image_list.append(image_path_dir + p)
    return image_list


def get_some_captions(caption_numbers, dataset="flickr30k"):
    captions = []
    if dataset == "flickr30k":
        name_list = get_f30k_name_list(IMAGE_PATH, True)
        image_caption_map = get_img_caption(IMAGE_CAPTIONS_FILE, dataset=dataset)
        images = name_list[:caption_numbers]
        for image in images:
            captions.append(image_caption_map[image].lower())
        return captions


def get_features(features_file):
    """
        input your features_file location and output a feature matrix
    Args:
        features_file:
    Returns:
         features_matrix
    """
    with open(features_file, "rb") as f:
        features = pickle.load(f)
    return features

captions = get_captions("/home/lemin/1TBdisk/PycharmProjects/ShowAttentionAndTell/data/annotations/captions_train2014.json")
create_vocabulary("/home/lemin/1TBdisk/PycharmProjects/ShowAttentionAndTell/data/vocab", captions, 25000)