import numpy as np

def mAP(pred, gt, n_class):
	AP = 0
	count = 0
	for i in range(n_class):
		conf = pred[:, i]
		conf = -conf
		gt_n = gt[:, i]
		sort_ind = np.argsort(conf)
		sort_arr = np.sort(conf)
		tp = np.zeros([pred.shape[0]])
		fp = np.zeros([pred.shape[0]])
		for rank, ind in enumerate(sort_ind):
			if gt_n[rank] == 1:
				tp[ind] = 1.
			else:
				fp[ind] = 1.
		if (tp == np.zeros(len(tp))).all():
			#print(tp)
			continue
		count = count + 1
		npos = np.sum(tp)
		tp_cum = np.cumsum(tp)
		fp_cum = np.cumsum(fp)
		rec = tp_cum / float(npos)
		prec = np.divide(tp_cum, tp_cum + fp_cum)
		tmp_ap = 0.
		for k, cont in enumerate(tp):
			if cont == 1.:
				tmp_ap = tmp_ap + prec[k]
		AP = AP + tmp_ap / np.sum(tp)
	if count == 0:
		return 0
	mAP = AP / float(count)
	return mAP

def hit_k(predict, label, k):
	total = len(predict)
	right = 0
	for id, sample in enumerate(predict):
		b = zip(sample, range(len(sample)))
		b.sort(key=lambda x: x[0],reverse=True)
		ranked_predict = [x[1] for x in b]
		for i in range(k):
			if label[id][ranked_predict[i]] == 1:
				right += 1
				break
	return float(right) / float(total)

