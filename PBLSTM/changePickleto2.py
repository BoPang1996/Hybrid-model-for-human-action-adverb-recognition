import pickle
import os

path = 'F:\\Lu_Lab\\video_dataset_definition\\LabelWork\\标注结果'
actions = os.listdir(path)
for action in actions:
	files = os.listdir(path + '/' + action)
	for file in files:
		print(path + '/' + action + '/' + file)
		with open(path + '/' + action + '/' + file, 'rb') as output:
			content = pickle.load(file=output)
		pickle.dump(content, open(path + '/' + action + '/' + file, 'wb'), protocol=2)
		print("finish " + ' ' + action + ' ' + file)
