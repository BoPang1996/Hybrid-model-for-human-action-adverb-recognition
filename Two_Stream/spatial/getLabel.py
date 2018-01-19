import numpy as np
import pickle
import os
import unicodedata

embedding_adverb = {
'hard':0,
'amazingly':1,
'wearily':2,
'clumsily':3,
'sweetly':4,
'excitedly':5,
'ironically':6,
'promptly':7,
'fast':8,
'kindly':9,
'carefully':10,
'seriously':11,
'barely':12,
'easily':13,
'slowly':14,
'quietly':15,
'precisely':16,
'gently':17,
'surprisedly':18,
'lightly':19,
'heavily':20,
'happily':21,
'freely':22,
'sadly':23,
'proudly':24,
'comfortably':25,
'calmly':26,
'vigorously':27,
'nervously':28,
'reluctantly':29,
'professionally':30,
'politely':31,
'painfully':32,
'angrily':33,
'patiently':34,
'bitterly':35,
'incidentally':36,
'frantically':37,
'intently':38,
'gracefully':39,
'flatly':40,
'confidently':41,
'weakly':42,
'solemnly':43,
'expertly':44,
'inexorably':45,
'triumphantly':46,
'hesitantly':47,
'dramatically':48,
'officially':49,
'anxiously':50
}
embedding_action = {
	'brush_hair':0,
	'chew':1,
	'clap':2,
	'climb_stairs':3,
	'dive':4,
	'draw_sword':5,
	'drink':6,
	'eat':7,
	'fall_floor':8,
	'hit':9,
	'hug':10,
	'kick':11,
	'kiss':12,
	'pick':13,
	'pour':14,
	'pullup':15,
	'punch':16,
	'push':17,
	'run':18,
	'shake_hands':19,
	'shoot_bow':20,
	'sit':21,
	'smoke':22,
	'stand':23,
	'swing_baseball':24,
	'sword':25,
	'sword_exercise':26,
	'talk':27,
	'throw':28,
	'walk':29,
	'wave':30,
	'shoot_gun': 31
}

def readLabel(sample, person, label_path):
	action = sample[0]
	label = []
	labelFiles = os.listdir(label_path + '/' + action)
	for label_file in labelFiles:
		with open(label_path + '/' + action + '/' + label_file, 'rb') as f:
			label_ = pickle.load(f)
			for i in label_:

				if i[0].encode('ascii','ignore') == sample[1][2:]:
				#if i[0] == sample[1][2:-7].decode():
					if int(person) + 1 >= len(i):
						continue
					for adv in i[int(person) + 1]:
						if adv == '-1':
							continue
						if embedding_adverb[adv] not in label:
							label.append(embedding_adverb[adv])


	adverb = np.zeros(len(embedding_adverb))

	for i in label:
		adverb[i] = 1
	act = np.zeros(len(embedding_action))
	act[embedding_action[action]] = 1
	return [act, adverb]
	# adverb = np.array(label)
	# act = np.array([embedding_action[action]])
	# return act, adverb

#print(readLabel(['brush_hair' ,'0_Silky_Straight_Hair_Original_brush_hair_h_nm_np1_ba_goo_1.avi.pickle'],'0_Silky_Straight_Hair_Original_brush_hair_h_nm_np1_ba_goo_1.avi.pickle'[0], 'F:\\Lu_Lab\\video_dataset_definition\\LabelWork\\LabelResult'))