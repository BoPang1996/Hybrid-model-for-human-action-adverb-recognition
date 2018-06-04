from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
import numpy as np
import os
import pickle
import tensorflow as tf
import random
import imageio

number_of_frame = 200
decay = 50

video_path = '/Disk8/HMDB/videos'
tracking_path = ''
pose_path = '/Disk8/HMDB/pose/'
out_path = 'Disk8/HMDB/pose_feature'

videoinfo_path = './control/videoinfo.pickle'
schedule_path = './control/schedule_extract'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if os.path.exists(videoinfo_path):
        with open(videoinfo_path, 'rb') as f1:
            print("Loading videoinfo")
            videoinfo = pickle.load(f1)
else:
    print("Getting Pickle of videoinfor")
    videoinfo = []
    actions = os.listdir(video_path)
    for action in actions:
        videos = os.listdir(video_path + '/' + action)
        for video in videos:
            videoinfo.append(action + '/' + video)
    with open(videoinfo_path, 'w') as f:
        pickle.dump(videoinfo,f)


# this could also be the output a different Keras model or layer
input_tensor = Input(shape=(299, 299, 3))  # this assumes K.image_data_format() == 'channels_last'
model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False, pooling='avg')


with open(schedule_path,"r") as f4:
    current = f4.readline()
    if int(current) == 0:
            print("Shufflling")
            random.shuffle(videoinfo)
        with open( videoinfo_path, 'wb') as f22:
        pickle.dump(videoinfo, f22)
for i in range(len(videoinfo)):
    if i < int(current):
        continue

    print ("Dealing with " + str(i))
    with open(tracking_path + '/' + videoinfo[i] + '.pickle', 'r') as f:
        tracking = pickle.load(f)
    N_person = len(tracking)
    features = []
    for person in range(N_person):
        features1 = []
        vid = imageio.get_reader(pose_path + '/' + 'Blending_' + videoinfo[i], 'ffmpeg')
        for frame_num in range(number_of_frame):
            if frame_num > len(vid):
                features1.append(np.zeros(feature[0].shape).tolist())
            else:
                img = vid.get_data(frame_num)
                bbox = tracking[person]['res'][frame_num]
                for a, h in enumerate(img):
                    for b, l in enumerate(h):
                        for c, _ in enumerate(l):
                            if a < bbox[1] or a > bbox[1] + bbox[3] or b < bbox[0] or b > bbox[0] + bbox[2]:
                                if img[a,b,c] > decay:
                                    img[a, b, c] = img[a,b,c] - decay
                                else:
                                    img[a, b, c] = 0
                x = cv2.resize(img,(299,299),interpolation=cv2.INTER_CUBIC)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)

                feature = model.predict(x)
                features1.append(feature[0].tolist())
        features.append(features1)

        features2 = []
        vid = imageio.get_reader(pose_path + '/' + 'No_blending_' + videoinfo[i], 'ffmpeg')
        for frame_num in range(number_of_frame):
            if frame_num > len(vid):
                features2.append(np.zeros(feature[0].shape).tolist())
            else:
                img = vid.get_data(frame_num)
                bbox = tracking[person]['res'][frame_num]
                for a, h in enumerate(img):
                    for b, l in enumerate(h):
                        for c, _ in enumerate(l):
                            if a < bbox[1] or a > bbox[1] + bbox[3] or b < bbox[0] or b > bbox[0] + bbox[2]
                                if img[a,b,c] > decay:
                                    img[a, b, c] = img[a,b,c] - decay
                                else:
                                    img[a, b, c] = 0
                x = cv2.resize(img,(299,299),interpolation=cv2.INTER_CUBIC)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)

                feature = model.predict(x)
                features2.append(feature[0].tolist())
        features.append(features2)


        features = np.array(features)

        if not os.path.exists(label_path + '/' + action):
            os.makedirs(label_path + '/' + action)
        with open( label_path + '/' + str(person) + '_' + videoinfo[i], 'wb') as f2:
            pickle.dump(features, f2)
            print(str(videoinfo[i][0]) + " Pickle Finished")

