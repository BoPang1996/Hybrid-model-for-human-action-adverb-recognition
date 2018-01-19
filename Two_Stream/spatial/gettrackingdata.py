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
from tqdm import tqdm
import math
import cv2

decay_ = 80



video_path = '/Disk8/HMDB/videos'
of_path = '/Disk8/HMDB/OpticalFlow/Result'
tracking_path = '/Disk8/HMDB/tracking/result'
out_path = '/Disk8/HMDB/Two_stream_input'

videoinfo_path = './control/videoinfo1.pickle'
schedule_path = './control/schedule_extract1'



if os.path.exists(videoinfo_path):
        with open(videoinfo_path, 'rb') as f1:
            print("Loading videoinfo")
            videoinfo = pickle.load(f1)
            print("all video: " + str(len(videoinfo)))
else:
    print("Getting Pickle of videoinfor")
    videoinfo = []
    actions = [ 'brush_hair', 'chew', 'clap', 'climb_stairs', 'dive', 'draw_sword', 'drink', 'eat', 'fall_floor', 'hit', 'hug', 'kick', 'kiss', 'pick', 'pour','pullup', 'punch', 'push', 'run', 'shake_hands', 'shoot_bow', 'shoot_gun', 'sit', 'smoke', 'stand', 'swing_baseball', 'sword', 'sword_exercise', 'talk', 'throw', 'walk', 'wave']
    n = 0
    for action in actions:
        samples = os.listdir(video_path + '/' + action)
        for sample in samples:
            videoinfo.append(action + '/' + sample)
    with open(videoinfo_path, 'w') as f:
        pickle.dump(videoinfo,f)




with open(schedule_path,"r") as f4:
    current = f4.readline()
    if int(current) == 0:
            print("Shufflling")
            random.shuffle(videoinfo)
            with open( videoinfo_path, 'wb') as f22:
                pickle.dump(videoinfo, f22)
for i in tqdm(range(len(videoinfo))):
    if i < int(current):
        continue

    #print ("Dealing with " + str(i))
    with open(tracking_path + '/' + videoinfo[i] + '.pickle', 'r') as f:
        tracking = pickle.load(f)
    N_person = len(tracking)
    for person in range(N_person):
        vid = imageio.get_reader(video_path + '/' + videoinfo[i].split('/')[0] + '/' + videoinfo[i].split('/')[1], 'ffmpeg')
        img =vid.get_data(int(len(vid)/2))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bbox = tracking[person]['res'][int(len(vid)/2)]
        for a, h in enumerate(img):
            for b, l in enumerate(h):
                if a < bbox[1] or a > bbox[1] + bbox[3] or b < bbox[0] or b > bbox[0] + bbox[2]:
                    decay = min(decay_, 1.5 * math.sqrt(
				        pow(max(0, abs(a - (bbox[1] * 2 + bbox[3]) / 2) - (bbox[3]) / 2), 2) + pow(
					        max(0, abs(b - (bbox[0] * 2 + bbox[2]) / 2) - (bbox[2]) / 2), 2)))
                    for c, _ in enumerate(l):
                        if img[a, b, c] > decay:
                            img[a, b, c] = img[a, b, c] - decay
                        else:
                            img[a, b, c] = 0
        x = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        if not os.path.exists(out_path + '/' + videoinfo[i].split('/')[0] + '/' + str(person) + '_' + videoinfo[i].split('/')[1] + '/' + 'rgb'):
            os.makedirs(out_path + '/' + videoinfo[i].split('/')[0] + '/' + str(person) + '_' + videoinfo[i].split('/')[1] + '/' + 'rgb')
        cv2.imwrite(out_path + '/' + videoinfo[i].split('/')[0] + '/' + str(person) + '_' + videoinfo[i].split('/')[1] + '/' + 'rgb' +'/' +  videoinfo[i].split('/')[1] + '.jpg',x)
        for frame_num in range(8*20):
            if frame_num % 8 != 0:
                continue
            if frame_num >= len(vid) or frame_num >= len(tracking[person]['res']):
                break
            else:
                if os.path.isfile(of_path + '/' + videoinfo[i].split('/')[0] + '/' + videoinfo[i].split('/')[1] + '/' + str(frame_num+1) + '_horizontal.jpg'):
                    img = cv2.imread(of_path + '/' + videoinfo[i].split('/')[0] + '/' + videoinfo[i].split('/')[1] + '/' + str(frame_num+1) + '_horizontal.jpg' )
                else:
                    print("No of: " + of_path + '/' + videoinfo[i].split('/')[0] + '/' + videoinfo[i].split('/')[1] + '/' + str(frame_num+1) + '_horizontal.jpg')
                    continue
                bbox = tracking[person]['res'][frame_num]
                for a, h in enumerate(img):
                    for b, l in enumerate(h):
                        if a < bbox[1] or a > bbox[1] + bbox[3] or b < bbox[0] or b > bbox[0] + bbox[2]:
                            decay = min(decay_, 1.5*math.sqrt(pow(max(0,abs(a-(bbox[1]*2 + bbox[3])/2)-(bbox[3])/2),2) + pow(max(0,abs(b-(bbox[0]*2 + bbox[2])/2)-(bbox[2])/2),2)))
                            for c, _ in enumerate(l):
                                if img[a,b,c] > decay:
                                    img[a, b, c] = img[a,b,c] - decay
                                else:
                                    img[a, b, c] = 0
                x = cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
                if not os.path.exists(out_path + '/' + videoinfo[i].split('/')[0] + '/' + str(person) + '_' + videoinfo[i].split('/')[1] + '/' + 'of'):
                    os.makedirs(out_path + '/' + videoinfo[i].split('/')[0] + '/' + str(person) + '_' + videoinfo[i].split('/')[1] + '/' + 'of')
                cv2.imwrite(out_path + '/' + videoinfo[i].split('/')[0] + '/' + str(person) + '_' + videoinfo[i].split('/')[1] + '/' + 'of' + '/' + str(frame_num) + '_horizontal.jpg', x)
                img = cv2.imread(
	                of_path + '/' + videoinfo[i].split('/')[0] + '/' + videoinfo[i].split('/')[1] + '/' + str(
		                frame_num + 1) + '_vertical.jpg')
                bbox = tracking[person]['res'][frame_num]
                for a, h in enumerate(img):
                    for b, l in enumerate(h):
                        if a < bbox[1] or a > bbox[1] + bbox[3] or b < bbox[0] or b > bbox[0] + bbox[2]:
                            decay = min(decay_, 1.5 * math.sqrt(
				                pow(max(0, abs(a - (bbox[1] * 2 + bbox[3]) / 2) - (bbox[3]) / 2), 2) + pow(
					                max(0, abs(b - (bbox[0] * 2 + bbox[2]) / 2) - (bbox[2]) / 2), 2)))
                            for c, _ in enumerate(l):
                                if img[a, b, c] > decay:
                                    img[a, b, c] = img[a, b, c] - decay
                                else:
                                    img[a, b, c] = 0
                x = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(out_path + '/' + videoinfo[i].split('/')[0] + '/' + str(person) + '_' + videoinfo[i].split('/')[
	                1] + '/' + 'of' + '/' + str(frame_num) + '_vertical.jpg', x)

        with open(schedule_path,"w") as f4:
            f4.write(str(i))
