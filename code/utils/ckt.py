import utils.data_utils as data_utils

captions = data_utils.get_captions("/home/lemin/1TBdisk/PycharmProjects/ShowAttentionAndTell/data/annotations/captions_train2014.json", "coco")

data_utils.create_vocabulary("/home/lemin/1TBdisk/PycharmProjects/ShowAttentionAndTell/data/vocab_25000", captions, 25000)