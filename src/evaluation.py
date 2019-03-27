import numpy as np

class evaluation(object):
    def __init__(self):
        pass

    def hit(self,gt_item, pred_items):
        if gt_item in pred_items:
            return 1
        return 0

    def ndcg(self,gt_item, pred_items):
        if gt_item in pred_items:
            index = np.where(pred_items == gt_item)[0][0]
            return np.reciprocal(np.log2(index+2))
        return 0
    

    def mrr(self,gt_item, pred_items):
        if gt_item in pred_items:
            index = np.where(pred_items == gt_item)[0][0]
            return np.reciprocal(float(index+1))
        else:
            return 0






