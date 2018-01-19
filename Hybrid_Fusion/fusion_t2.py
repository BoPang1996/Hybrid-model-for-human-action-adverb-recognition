import pickle

PBLSTM_result_path = None
TWO_STREAM_result_path = None

adverb_label = pickle.load(open(PBLSTM_result_path + '/' + 'adverb_label.pickle', 'rb'))

PBLSEM_result_adverb = pickle.load open(PBLSTM_result_path + '/' + 'adverb_result.pickle', 'rb'))

two_stream_result_adverb = pickle.load open(TWO_STREAM_result_path + '/' + 'adverb_result.pickle', 'rb'))

preds_adv = np.divide((PBLSEM_result_adverb + two_stream_result_adverb), 2)


mAP_adv = mAP(np.array(preds_adv), np.array(adverb_label), 51)
prec1_adv = hit_k(np.array(preds_adv), np.array(adverb_label), 1)
prec5_adv = hit_k(np.array(preds_adv), np.array(adverb_label), 5)


print("Total map_adverb: " + str(mAP_adv) + ' | hit_1_adverb: ' + str(prec1_adv) + ' | hit_5_adverb: ' + str(prec5_adv) )
