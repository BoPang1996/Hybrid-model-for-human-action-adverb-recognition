import pickle

PBLSTM_result_path = None
TWO_STREAM_result_path = None

action_label = pickle.load(open(PBLSTM_result_path + '/' + 'action_label.pickle', 'rb'))
adverb_label = pickle.load(open(PBLSTM_result_path + '/' + 'adverb_label.pickle', 'rb'))

PBLSEM_result_action = pickle.load open(PBLSTM_result_path + '/' + 'action_result.pickle', 'rb'))
PBLSEM_result_adverb = pickle.load open(PBLSTM_result_path + '/' + 'adverb_result.pickle', 'rb'))

two_stream_result_action = pickle.load open(TWO_STREAM_result_path + '/' + 'action_result.pickle', 'rb'))
two_stream_result_adverb = pickle.load open(TWO_STREAM_result_path + '/' + 'adverb_result.pickle', 'rb'))

preds_act = np.divide((PBLSEM_result_action + two_stream_result_action), 2)
preds_adv = np.divide((PBLSEM_result_adverb + two_stream_result_adverb), 2)


mAP_adv = mAP(np.array(preds_adv), np.array(adverb_label), 51)
prec1_adv = hit_k(np.array(preds_adv), np.array(adverb_label), 1)
prec5_adv = hit_k(np.array(preds_adv), np.array(adverb_label), 5)

mAP_act = mAP(np.array(preds_act), np.array(action_label), 51)
prec1_act = hit_k(np.array(preds_act), np.array(action_label), 1)
prec5_act = hit_k(np.array(preds_act), np.array(action_label), 5)

print("Total mAP_action: " + str(mAP_act) + " | map_adverb: " + str(mAP_adv) + " | hit_1_action: " + str(prec1_act) + ' | hit_1_adverb: ' + str(prec1_adv) + " | hit_5_action: " + str(prec5_act) + ' | hit_5_adverb: ' + str(prec5_adv) )
