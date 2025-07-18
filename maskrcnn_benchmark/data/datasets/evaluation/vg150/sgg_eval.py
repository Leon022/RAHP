import numpy as np
from functools import reduce
import torch
from torchvision.ops.boxes import box_area
import copy
from collections import defaultdict
from sklearn import metrics
from maskrcnn_benchmark.utils.miscellaneous import intersect_2d, argsort_desc, bbox_overlaps
from abc import ABC, abstractmethod
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

topk_range = [20, 50, 100, 200, 300, 500]

def intersect_2d(x1, x2):
    """
    Given two arrays [m1, n], [m2,n], returns a [m1, m2] array where each entry is True if those
    rows match.
    :param x1: [m1, n] numpy array
    :param x2: [m2, n] numpy array
    :return: [m1, m2] bool array of the intersections
    """
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("Input arrays must have same #columns")

    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    res = (x1[..., None] == x2.T[None, ...]).all(1)
    return res

def argsort_desc(scores):
    """
    Returns the indices that sort scores descending in a smart way
    :param scores: Numpy array of arbitrary size
    :return: an array of size [numel(scores), dim(scores)] where each row is the index you'd
             need to get the score.
    """
    return np.column_stack(np.unravel_index(np.argsort(-scores.ravel()), scores.shape))

def bbox_overlaps(boxes1, boxes2):
    """
    Parameters:
        boxes1 (m, 4) [List or np.array] : bounding boxes of (x1,y1,x2,y2)
        boxes2 (n, 4) [List or np.array] : bounding boxes of (x1,y1,x2,y2)
    Return:
        iou (m, n) [np.array]
    """
    boxes1 = BoxList(boxes1, (0, 0), 'xyxy')
    boxes2 = BoxList(boxes2, (0, 0), 'xyxy')
    iou = boxlist_iou(boxes1, boxes2).cpu().numpy()
    return iou


class SceneGraphEvaluation(ABC):
    def __init__(self, result_dict):
        super().__init__()
        self.result_dict = result_dict
 
    @abstractmethod
    def register_container(self, mode):
        print("Register Result Container")
        pass
    
    @abstractmethod
    def generate_print_string(self, mode):
        print("Generate Print String")
        pass


"""
Traditional Recall, implement based on:
https://github.com/rowanz/neural-motifs
"""
class SGRecall(SceneGraphEvaluation):
    def __init__(self, result_dict, num_rel):
        super(SGRecall, self).__init__(result_dict)
        self.num_rel = num_rel
        self.type = "recall"

    def register_container(self, mode):
        self.result_dict[mode + '_recall'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_recall'].items():
            result_str += '    R @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=Recall(Main).' % mode
        result_str += '\n'
        return result_str

    def calculate_recall(self, global_container, local_container, mode):
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        pred_rels_cls = local_container['pred_rel_labels']
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        gt_boxes = local_container['gt_boxes']
        pred_classes = local_container['pred_classes']
        pred_boxes = local_container['pred_boxes']
        obj_scores = local_container['obj_scores']

        iou_thres = global_container['iou_thres']

        # directly use the rel labels produced by the post procs
        # ce loss
        pred_rels = np.column_stack((pred_rel_inds, 1+rel_scores[:,1:].argmax(1)))
        pred_scores = rel_scores[:,1:].max(1)
        
        # focal loss
        # pred_rels = np.column_stack((pred_rel_inds, pred_rels_cls))
        # pred_scores = rel_scores[:,:].max(1)

        gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)
        local_container['gt_triplets'] = gt_triplets
        local_container['gt_triplet_boxes'] = gt_triplet_boxes
        pred_triplets, pred_triplet_boxes, pred_triplet_scores = _triplet(
                pred_rels, pred_classes, pred_boxes, pred_scores, obj_scores)

        # Compute recall. It's most efficient to match once and then do recall after
        pred_to_gt = _compute_pred_matches(
            gt_triplets,
            pred_triplets,
            gt_triplet_boxes,
            pred_triplet_boxes,
            iou_thres,
            phrdet=mode=='phrdet',
        )
        local_container['pred_to_gt'] = pred_to_gt

        for k in self.result_dict[mode + '_recall']:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k])
            rec_i = float(len(match)) / float(gt_rels.shape[0])
            self.result_dict[mode + '_recall'][k].append(rec_i)

        return local_container
"""
No Graph Constraint Recall, implement based on:
https://github.com/rowanz/neural-motifs
"""
class SGNoGraphConstraintRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGNoGraphConstraintRecall, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict[mode + '_recall_nogc'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_recall_nogc'].items():
            result_str += ' ng-R @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=No Graph Constraint Recall(Main).' % mode
        result_str += '\n'
        return result_str

    def calculate_recall(self, global_container, local_container, mode):
        obj_scores = local_container['obj_scores']
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        pred_boxes = local_container['pred_boxes']
        pred_classes = local_container['pred_classes']
        gt_rels = local_container['gt_rels']

        obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
        nogc_overall_scores = obj_scores_per_rel[:,None] * rel_scores[:,1:]
        nogc_score_inds = argsort_desc(nogc_overall_scores)[:100]
        nogc_pred_rels = np.column_stack((pred_rel_inds[nogc_score_inds[:,0]], nogc_score_inds[:,1]+1))
        nogc_pred_scores = rel_scores[nogc_score_inds[:,0], nogc_score_inds[:,1]+1]

        nogc_pred_triplets, nogc_pred_triplet_boxes, _ = _triplet(
                nogc_pred_rels, pred_classes, pred_boxes, nogc_pred_scores, obj_scores
        )

        # No Graph Constraint
        gt_triplets = local_container['gt_triplets']
        gt_triplet_boxes = local_container['gt_triplet_boxes']
        iou_thres = global_container['iou_thres']

        nogc_pred_to_gt = _compute_pred_matches(
            gt_triplets,
            nogc_pred_triplets,
            gt_triplet_boxes,
            nogc_pred_triplet_boxes,
            iou_thres,
            phrdet=mode=='phrdet',
        )

        local_container['nogc_pred_to_gt'] = nogc_pred_to_gt

        for k in self.result_dict[mode + '_recall_nogc']:
            match = reduce(np.union1d, nogc_pred_to_gt[:k])
            rec_i = float(len(match)) / float(gt_rels.shape[0])
            self.result_dict[mode + '_recall_nogc'][k].append(rec_i)

        return local_container

"""
Zero Shot Scene Graph
Only calculate the triplet that not occurred in the training set

Modified for: 
He, Tao, et al. "Towards Open-vocabulary Scene Graph Generation with Prompt-based Finetuning." ECCV2022.
"""
class SGZeroShotRecall(SceneGraphEvaluation):
    def __init__(self, result_dict, unseen_obj_cats, unseen_predicate_cars):
        super(SGZeroShotRecall, self).__init__(result_dict)
        self.novel_count = 0
        self.all_count = 0
        self.unseen_obj_cats = unseen_obj_cats
        self.unseen_predicate_cars = unseen_predicate_cars

    def register_container(self, mode):
        self.result_dict[mode + '_zeroshot_recall'] = {20: [], 50: [], 100: []}
        self.result_dict[mode + '_seenshot_recall'] = {20: [], 50: [], 100: []}
        self.result_dict[mode + '_seen_pair_seen_pred_recall'] = {20: [], 50: [], 100: []}
        self.result_dict[mode + '_seen_pair_unseen_pred_recall'] = {20: [], 50: [], 100: []}
        self.result_dict[mode + '_unseen_pair_seen_pred_recall'] = {20: [], 50: [], 100: []}
        self.result_dict[mode + '_unseen_pair_unseen_pred_recall'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_zeroshot_recall'].items():
            result_str += '   zR @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=Zero Shot Recall.\n' % mode

        result_str += 'SGG eval: '
        for k, v in self.result_dict[mode + '_seenshot_recall'].items():
            result_str += '   sR @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=Seen Shot Recall.\n' % mode
        # result_str += f'all/novel={self.all_count}/{self.novel_count}\n'

        result_str += 'SGG eval: '
        for k, v in self.result_dict[mode + '_seen_pair_seen_pred_recall'].items():
            result_str += '   sR @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=Seen Pair Seen Pred Recall.\n' % mode
        result_str += 'SGG eval: '
        for k, v in self.result_dict[mode + '_seen_pair_unseen_pred_recall'].items():
            result_str += '   sR @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=Seen Pair Unseen Pred Recall.\n' % mode
        result_str += 'SGG eval: '
        for k, v in self.result_dict[mode + '_unseen_pair_seen_pred_recall'].items():
            result_str += '   sR @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=Unseen Pair Seen Pred Recall.\n' % mode
        result_str += 'SGG eval: '
        for k, v in self.result_dict[mode + '_unseen_pair_unseen_pred_recall'].items():
            result_str += '   sR @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=Unseen Pair Unseen Pred Recall.\n' % mode

        return result_str

    def prepare_zeroshot(self, global_container, local_container):
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        # zeroshot_triplets = global_container['zeroshot_triplet']

        sub_id, ob_id, pred_label = gt_rels[:, 0], gt_rels[:, 1], gt_rels[:, 2]
        gt_triplets = np.column_stack((gt_classes[sub_id], gt_classes[ob_id], pred_label))  # num_rel, 3

        # self.zeroshot_idx = np.where( intersect_2d(gt_triplets, zeroshot_triplets).sum(-1) > 0 )[0].tolist()
        # self.zeroshot_idx = [] # both subj & obj in unseen set
        # for i, (s, o, p) in enumerate(gt_triplets):
        #     # if s not in self.unseen_obj_cats and o not in self.unseen_obj_cats:
        #     if s in self.unseen_obj_cats and o in self.unseen_obj_cats:
        #         self.zeroshot_idx.append(i)
        #         self.novel_count += 1
        #     self.all_count += 1
        self.zeroshot_idx = [] # both subj & obj in unseen set
        self.seenshot_idx = [] # both subj & obj in seen set
        self.obj_zero_pred_seen_idx = [] # both subj & obj in unseen set
        self.obj_seen_pred_zero_idx = [] # both subj & obj in seen set
        """split seen pair and unseen pair"""
        self.seen_pair_seen_pred = []
        self.seen_pair_unseen_pred = []
        self.unseen_pair_seen_pred = []
        self.unseen_pair_unseen_pred = []
        for i, (s, o, p) in enumerate(gt_triplets):
            # if s not in self.unseen_obj_cats and o not in self.unseen_obj_cats:
            if p-1 in self.unseen_predicate_cars:
                self.zeroshot_idx.append(i)
                self.novel_count += 1
            else:
                self.seenshot_idx.append(i)
            
            if o in self.unseen_obj_cats and s in self.unseen_obj_cats:
                self.obj_zero_pred_seen_idx.append(i)
            if o not in self.unseen_obj_cats and s not in self.unseen_obj_cats:
                self.obj_seen_pred_zero_idx.append(i)
            self.all_count += 1

            if (s,o) in self.base_pair_labels_all and p-1 not in self.unseen_predicate_cars:
                self.seen_pair_seen_pred.append(i)
            elif (s,o) in self.base_pair_labels_all and p-1 in self.unseen_predicate_cars:
                self.seen_pair_unseen_pred.append(i)
            elif not (s,o) in self.base_pair_labels_all and p-1 not in self.unseen_predicate_cars:
                self.unseen_pair_seen_pred.append(i)
            else:
                self.unseen_pair_unseen_pred.append(i)

    def calculate_recall(self, global_container, local_container, mode):
        pred_to_gt = local_container['pred_to_gt']

        for k in self.result_dict[mode + '_zeroshot_recall']:
            # Zero Shot Recall
            match = reduce(np.union1d, pred_to_gt[:k])
            if len(self.zeroshot_idx) > 0:
                if not isinstance(match, (list, tuple)):
                    match_list = match.tolist()
                else:
                    match_list = match
                zeroshot_match = len(self.zeroshot_idx) + len(match_list) - len(set(self.zeroshot_idx + match_list))
                zero_rec_i = float(zeroshot_match) / float(len(self.zeroshot_idx))
                self.result_dict[mode + '_zeroshot_recall'][k].append(zero_rec_i)

        for k in self.result_dict[mode + '_seenshot_recall']:
            # Zero Shot Recall
            match = reduce(np.union1d, pred_to_gt[:k])
            if len(self.seenshot_idx) > 0:
                if not isinstance(match, (list, tuple)):
                    match_list = match.tolist()
                else:
                    match_list = match
                zeroshot_match = len(self.seenshot_idx) + len(match_list) - len(set(self.seenshot_idx + match_list))
                zero_rec_i = float(zeroshot_match) / float(len(self.seenshot_idx))
                self.result_dict[mode + '_seenshot_recall'][k].append(zero_rec_i)
        
        for k in self.result_dict[mode + '_seen_pair_seen_pred_recall']:
            # Zero Shot Recall
            match = reduce(np.union1d, pred_to_gt[:k])
            if len(self.seen_pair_seen_pred) > 0:
                if not isinstance(match, (list, tuple)):
                    match_list = match.tolist()
                else:
                    match_list = match
                zeroshot_match = len(self.seen_pair_seen_pred) + len(match_list) - len(set(self.seen_pair_seen_pred + match_list))
                zero_rec_i = float(zeroshot_match) / float(len(self.seen_pair_seen_pred))
                self.result_dict[mode + '_seen_pair_seen_pred_recall'][k].append(zero_rec_i)
        
        for k in self.result_dict[mode + '_seen_pair_unseen_pred_recall']:
            # Zero Shot Recall
            match = reduce(np.union1d, pred_to_gt[:k])
            if len(self.seen_pair_unseen_pred) > 0:
                if not isinstance(match, (list, tuple)):
                    match_list = match.tolist()
                else:
                    match_list = match
                zeroshot_match = len(self.seen_pair_unseen_pred) + len(match_list) - len(set(self.seen_pair_unseen_pred + match_list))
                zero_rec_i = float(zeroshot_match) / float(len(self.seen_pair_unseen_pred))
                self.result_dict[mode + '_seen_pair_unseen_pred_recall'][k].append(zero_rec_i)
        
        for k in self.result_dict[mode + '_unseen_pair_seen_pred_recall']:
            # Zero Shot Recall
            match = reduce(np.union1d, pred_to_gt[:k])
            if len(self.unseen_pair_seen_pred) > 0:
                if not isinstance(match, (list, tuple)):
                    match_list = match.tolist()
                else:
                    match_list = match
                zeroshot_match = len(self.unseen_pair_seen_pred) + len(match_list) - len(set(self.unseen_pair_seen_pred + match_list))
                zero_rec_i = float(zeroshot_match) / float(len(self.unseen_pair_seen_pred))
                self.result_dict[mode + '_unseen_pair_seen_pred_recall'][k].append(zero_rec_i)
        
        for k in self.result_dict[mode + '_unseen_pair_unseen_pred_recall']:
            # Zero Shot Recall
            match = reduce(np.union1d, pred_to_gt[:k])
            if len(self.unseen_pair_unseen_pred) > 0:
                if not isinstance(match, (list, tuple)):
                    match_list = match.tolist()
                else:
                    match_list = match
                zeroshot_match = len(self.unseen_pair_unseen_pred) + len(match_list) - len(set(self.unseen_pair_unseen_pred + match_list))
                zero_rec_i = float(zeroshot_match) / float(len(self.unseen_pair_unseen_pred))
                self.result_dict[mode + '_unseen_pair_unseen_pred_recall'][k].append(zero_rec_i)

"""
No Graph Constraint Mean Recall
"""
class SGNGZeroShotRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGNGZeroShotRecall, self).__init__(result_dict)
    
    def register_container(self, mode):
        self.result_dict[mode + '_ng_zeroshot_recall'] = {20: [], 50: [], 100: []} 

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_ng_zeroshot_recall'].items():
            result_str += 'ng-zR @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=No Graph Constraint Zero Shot Recall.' % mode
        result_str += '\n'
        return result_str

    def prepare_zeroshot(self, global_container, local_container):
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        zeroshot_triplets = global_container['zeroshot_triplet']

        sub_id, ob_id, pred_label = gt_rels[:, 0], gt_rels[:, 1], gt_rels[:, 2]
        gt_triplets = np.column_stack((gt_classes[sub_id], gt_classes[ob_id], pred_label))  # num_rel, 3

        self.zeroshot_idx = np.where( intersect_2d(gt_triplets, zeroshot_triplets).sum(-1) > 0 )[0].tolist()

    def calculate_recall(self, global_container, local_container, mode):
        pred_to_gt = local_container['nogc_pred_to_gt']

        for k in self.result_dict[mode + '_ng_zeroshot_recall']:
            # Zero Shot Recall
            match = reduce(np.union1d, pred_to_gt[:k])
            if len(self.zeroshot_idx) > 0:
                if not isinstance(match, (list, tuple)):
                    match_list = match.tolist()
                else:
                    match_list = match
                zeroshot_match = len(self.zeroshot_idx) + len(match_list) - len(set(self.zeroshot_idx + match_list))
                zero_rec_i = float(zeroshot_match) / float(len(self.zeroshot_idx))
                self.result_dict[mode + '_ng_zeroshot_recall'][k].append(zero_rec_i)


"""
Give Ground Truth Object-Subject Pairs
Calculate Recall for SG-Cls and Pred-Cls
Only used in https://github.com/NVIDIA/ContrastiveLosses4VRD for sgcls and predcls
"""
class SGPairAccuracy(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGPairAccuracy, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict[mode + '_accuracy_hit'] = {20: [], 50: [], 100: []}
        self.result_dict[mode + '_accuracy_count'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_accuracy_hit'].items():
            a_hit = np.mean(v)
            a_count = np.mean(self.result_dict[mode + '_accuracy_count'][k])
            result_str += '    A @ %d: %.4f; ' % (k, a_hit/a_count)
        result_str += ' for mode=%s, type=TopK Accuracy.' % mode
        result_str += '\n'
        return result_str

    def prepare_gtpair(self, local_container):
        pred_pair_idx = local_container['pred_rel_inds'][:, 0] * 1024 + local_container['pred_rel_inds'][:, 1]
        gt_pair_idx = local_container['gt_rels'][:, 0] * 1024 + local_container['gt_rels'][:, 1]
        self.pred_pair_in_gt = (pred_pair_idx[:, None] == gt_pair_idx[None, :]).sum(-1) > 0

    def calculate_recall(self, global_container, local_container, mode):
        pred_to_gt = local_container['pred_to_gt']
        gt_rels = local_container['gt_rels']

        for k in self.result_dict[mode + '_accuracy_hit']:
            # to calculate accuracy, only consider those gt pairs
            # This metric is used by "Graphical Contrastive Losses for Scene Graph Parsing" 
            # for sgcls and predcls
            if mode != 'sgdet':
                gt_pair_pred_to_gt = []
                for p, flag in zip(pred_to_gt, self.pred_pair_in_gt):
                    if flag:
                        gt_pair_pred_to_gt.append(p)
                if len(gt_pair_pred_to_gt) > 0:
                    gt_pair_match = reduce(np.union1d, gt_pair_pred_to_gt[:k])
                else:
                    gt_pair_match = []
                self.result_dict[mode + '_accuracy_hit'][k].append(float(len(gt_pair_match)))
                self.result_dict[mode + '_accuracy_count'][k].append(float(gt_rels.shape[0]))


"""
Mean Recall: Proposed in:
https://arxiv.org/pdf/1812.01880.pdf CVPR, 2019
"""
class SGMeanRecall(SceneGraphEvaluation):
    def __init__(self, result_dict, num_rel, ind_to_predicates, unseen_predicate_cars, print_detail=False):
        super(SGMeanRecall, self).__init__(result_dict)
        self.num_rel = num_rel
        self.print_detail = print_detail
        self.rel_name_list = ind_to_predicates[1:] # remove __background__
        self.type = "mean_recall"
        # cacao
        self.unseen_pred_cats = unseen_predicate_cars
        self.seen_pred_cats = [i for i in range(51) if i not in unseen_predicate_cars]

    def register_container(self, mode):
        #self.result_dict[mode + '_recall_hit'] = {20: [0]*self.num_rel, 50: [0]*self.num_rel, 100: [0]*self.num_rel}
        #self.result_dict[mode + '_recall_count'] = {20: [0]*self.num_rel, 50: [0]*self.num_rel, 100: [0]*self.num_rel}
        self.result_dict[mode + '_mean_recall'] = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_dict[mode + '_mean_recall_collect'] = {20: [[] for i in range(self.num_rel)], 50: [[] for i in range(self.num_rel)], 100: [[] for i in range(self.num_rel)]}
        self.result_dict[mode + '_mean_recall_list'] = {20: [], 50: [], 100: []}

        self.result_dict[mode + '_mean_recall_base'] = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_dict[mode + '_mean_recall_list_base'] = {20: [], 50: [], 100: []}
        self.result_dict[mode + '_mean_recall_novel'] = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_dict[mode + '_mean_recall_list_novel'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_mean_recall'].items():
            result_str += '   mR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=Mean Recall.' % mode
        result_str += '\n'

        for k, v in self.result_dict[mode + '_mean_recall_base'].items():
            result_str += '   mR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=Mean Recall of base.' % mode
        result_str += '\n'
        for k, v in self.result_dict[mode + '_mean_recall_novel'].items():
            result_str += '   mR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=Mean Recall of novel.' % mode
        result_str += '\n'

        if self.print_detail:
            result_str += '----------------------- Details ------------------------\n'
            for n, r in zip(self.rel_name_list, self.result_dict[mode + '_mean_recall_list'][100]):
                result_str += '({}:{:.4f}) '.format(str(n), r)
            result_str += '\n'
            result_str += '--------------------------------------------------------\n'

        return result_str

    def collect_mean_recall_items(self, global_container, local_container, mode):
        pred_to_gt = local_container['pred_to_gt']
        gt_rels = local_container['gt_rels']

        for k in self.result_dict[mode + '_mean_recall_collect']:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k])
            # NOTE: by kaihua, calculate Mean Recall for each category independently
            # this metric is proposed by: CVPR 2019 oral paper "Learning to Compose Dynamic Tree Structures for Visual Contexts"
            recall_hit = [0] * self.num_rel
            recall_count = [0] * self.num_rel
            for idx in range(gt_rels.shape[0]):
                local_label = gt_rels[idx,2]
                recall_count[int(local_label)] += 1
                recall_count[0] += 1

            for idx in range(len(match)):
                local_label = gt_rels[int(match[idx]),2]
                recall_hit[int(local_label)] += 1
                recall_hit[0] += 1
            
            for n in range(self.num_rel):
                if recall_count[n] > 0:
                    self.result_dict[mode + '_mean_recall_collect'][k][n].append(float(recall_hit[n] / recall_count[n]))
 

    def calculate_mean_recall(self, mode):
        for k, v in self.result_dict[mode + '_mean_recall'].items():
            sum_recall = 0
            num_rel_no_bg = self.num_rel - 1
            for idx in range(num_rel_no_bg):
                if len(self.result_dict[mode + '_mean_recall_collect'][k][idx+1]) == 0:
                    tmp_recall = 0.0
                else:
                    tmp_recall = np.mean(self.result_dict[mode + '_mean_recall_collect'][k][idx+1])
                self.result_dict[mode + '_mean_recall_list'][k].append(tmp_recall)
                sum_recall += tmp_recall

            self.result_dict[mode + '_mean_recall'][k] = sum_recall / float(num_rel_no_bg)

            sum_recall_base = 0
            for idx in self.seen_pred_cats:
                if len(self.result_dict[mode + '_mean_recall_collect'][k][idx+1]) == 0:
                    tmp_recall = 0.0
                else:
                    tmp_recall = np.mean(self.result_dict[mode + '_mean_recall_collect'][k][idx+1])
                self.result_dict[mode + '_mean_recall_list_base'][k].append(tmp_recall)
                sum_recall_base += tmp_recall

            self.result_dict[mode + '_mean_recall_base'][k] = sum_recall_base / float(len(self.seen_pred_cats))

            sum_recall_novel = 0
            for idx in self.unseen_pred_cats:
                if len(self.result_dict[mode + '_mean_recall_collect'][k][idx+1]) == 0:
                    tmp_recall = 0.0
                else:
                    tmp_recall = np.mean(self.result_dict[mode + '_mean_recall_collect'][k][idx+1])
                self.result_dict[mode + '_mean_recall_list_novel'][k].append(tmp_recall)
                sum_recall_novel += tmp_recall

            self.result_dict[mode + '_mean_recall_novel'][k] = sum_recall_novel / float(len(self.unseen_pred_cats))

        return

    def eval_given_cate_set_perf(self, mode, cate_list):
        selected_cate_mR = {}
        for k, v in self.result_dict[mode + "_mean_recall"].items():
            sum_recall = 0
            for idx in cate_list:
                tmp_recall = self.result_dict[mode + "_mean_recall_list"][k][idx]  # start from 0
                sum_recall += tmp_recall
            selected_cate_mR[k] = sum_recall / float(len(cate_list))
        
        result_str = "selected categories set mR eval: "
        for k, v in selected_cate_mR.items():
            result_str += " mR @ %d: %.4f; " % (k, float(v))
        result_str += " for mode=%s, type=Mean Recall." % mode
        result_str += "\n"
        if self.print_detail:
            for k in topk_range: 
                result_str += f"Per-class recall@{k}: \n"
                for cate_id in cate_list:
                    result_str += f"({self.rel_name_list[cate_id]}:{self.result_dict[mode + '_mean_recall_list'][k][cate_id]:.4f}) "
                result_str += "\n"
            result_str += "\n"

        return result_str

"""
No Graph Constraint Mean Recall
"""
class SGNGMeanRecall(SceneGraphEvaluation):
    def __init__(self, result_dict, num_rel, ind_to_predicates, print_detail=False):
        super(SGNGMeanRecall, self).__init__(result_dict)
        self.num_rel = num_rel
        self.print_detail = print_detail
        self.rel_name_list = ind_to_predicates[1:] # remove __background__

    def register_container(self, mode):
        self.result_dict[mode + '_ng_mean_recall'] = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_dict[mode + '_ng_mean_recall_collect'] = {20: [[] for i in range(self.num_rel)], 50: [[] for i in range(self.num_rel)], 100: [[] for i in range(self.num_rel)]}
        self.result_dict[mode + '_ng_mean_recall_list'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_ng_mean_recall'].items():
            result_str += 'ng-mR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=No Graph Constraint Mean Recall.' % mode
        result_str += '\n'
        if self.print_detail:
            result_str += '----------------------- Details ------------------------\n'
            for n, r in zip(self.rel_name_list, self.result_dict[mode + '_ng_mean_recall_list'][100]):
                result_str += '({}:{:.4f}) '.format(str(n), r)
            result_str += '\n'
            result_str += '--------------------------------------------------------\n'

        return result_str

    def collect_mean_recall_items(self, global_container, local_container, mode):
        pred_to_gt = local_container['nogc_pred_to_gt']
        gt_rels = local_container['gt_rels']

        for k in self.result_dict[mode + '_ng_mean_recall_collect']:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k])
            # NOTE: by kaihua, calculate Mean Recall for each category independently
            # this metric is proposed by: CVPR 2019 oral paper "Learning to Compose Dynamic Tree Structures for Visual Contexts"
            recall_hit = [0] * self.num_rel
            recall_count = [0] * self.num_rel
            for idx in range(gt_rels.shape[0]):
                local_label = gt_rels[idx,2]
                recall_count[int(local_label)] += 1
                recall_count[0] += 1

            for idx in range(len(match)):
                local_label = gt_rels[int(match[idx]),2]
                recall_hit[int(local_label)] += 1
                recall_hit[0] += 1
            
            for n in range(self.num_rel):
                if recall_count[n] > 0:
                    self.result_dict[mode + '_ng_mean_recall_collect'][k][n].append(float(recall_hit[n] / recall_count[n]))
 

    def calculate_mean_recall(self, mode):
        for k, v in self.result_dict[mode + '_ng_mean_recall'].items():
            sum_recall = 0
            num_rel_no_bg = self.num_rel - 1
            for idx in range(num_rel_no_bg):
                if len(self.result_dict[mode + '_ng_mean_recall_collect'][k][idx+1]) == 0:
                    tmp_recall = 0.0
                else:
                    tmp_recall = np.mean(self.result_dict[mode + '_ng_mean_recall_collect'][k][idx+1])
                self.result_dict[mode + '_ng_mean_recall_list'][k].append(tmp_recall)
                sum_recall += tmp_recall

            self.result_dict[mode + '_ng_mean_recall'][k] = sum_recall / float(num_rel_no_bg)
        return

"""
Accumulate Recall:
calculate recall on the whole dataset instead of each image
"""
class SGAccumulateRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGAccumulateRecall, self).__init__(result_dict)

    def register_container(self, mode):
        self.result_dict[mode + '_accumulate_recall'] = {20: 0.0, 50: 0.0, 100: 0.0}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_accumulate_recall'].items():
            result_str += '   aR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=Accumulate Recall.' % mode
        result_str += '\n'
        return result_str

    def calculate_accumulate(self, mode):
        for k, v in self.result_dict[mode + '_accumulate_recall'].items():
            self.result_dict[mode + '_accumulate_recall'][k] = float(self.result_dict[mode + '_recall_hit'][k][0]) / float(self.result_dict[mode + '_recall_count'][k][0] + 1e-10)

        return 


def _triplet(relations, classes, boxes, predicate_scores=None, class_scores=None):
    """
    format relations of (sub_id, ob_id, pred_label) into triplets of (sub_label, pred_label, ob_label)
    Parameters:
        relations (#rel, 3) : (sub_id, ob_id, pred_label)
        classes (#objs, ) : class labels of objects
        boxes (#objs, 4)
        predicate_scores (#rel, ) : scores for each predicate
        class_scores (#objs, ) : scores for each object
    Returns: 
        triplets (#rel, 3) : (sub_label, pred_label, ob_label)
        triplets_boxes (#rel, 8) array of boxes for the parts
        triplets_scores (#rel, 3) : (sub_score, pred_score, ob_score)
    """
    sub_id, ob_id, pred_label = relations[:, 0], relations[:, 1], relations[:, 2]
    triplets = np.column_stack((classes[sub_id], pred_label, classes[ob_id]))
    triplet_boxes = np.column_stack((boxes[sub_id], boxes[ob_id]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[sub_id], predicate_scores, class_scores[ob_id],
        ))

    return triplets, triplet_boxes, triplet_scores


def _compute_pred_matches(gt_triplets, pred_triplets,
                 gt_boxes, pred_boxes, iou_thres, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    Return:
        pred_to_gt [List of List]
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:,:2], box_union.max(1)[:,2:]), 1)

            inds = bbox_overlaps(gt_box_union[None], box_union)[0] >= iou_thres

        else:
            sub_iou = bbox_overlaps(gt_box[None,:4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None,4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thres) & (obj_iou >= iou_thres)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt

class SGRelVecRecall(SceneGraphEvaluation):
    def __init__(self, cfg, result_dict, ind_to_predicates):
        super(SGRelVecRecall, self).__init__(result_dict)
        self.type = "rel_vec_recall"
        self.cfg = cfg
        self.ind_to_predicates = ind_to_predicates
        self.num_rel_cls = len(ind_to_predicates)


    def register_container(self, mode):
        for match_type in ['det', 'loc']:
            self.result_dict[f"{mode}_{self.type}_{match_type}"] = { k: [] for k in topk_range} 
            self.result_dict[f"{mode}_{match_type}_mean_recall_collect"] = {
                k: torch.zeros((self.num_rel_cls, 2), dtype=torch.int64)
                for k in topk_range
            }

    def generate_print_string(self, mode):

        result_str = "\nSGG rel vector eval: "
        for match_type in ['det', 'loc']:
            for k, v in self.result_dict[f"{mode}_{self.type}_{match_type}"].items():
                result_str += "  R @ %d: %.4f; " % (k, np.mean(v))
            result_str += f" for mode={mode}, type=Recall({match_type})." 
            result_str += "\n"

        result_str += "Per-class: \n"
        for match_type in ['det', 'loc']:
            for k, v in self.result_dict[f"{mode}_{match_type}_mean_recall_collect"].items():
                result_str += "  mR @ %d: %.4f; " % (k, torch.mean(v[1:,0] / (v[1:,1] + 1e-5)))
            result_str += f" for mode={mode}, type=Recall({match_type})." 
            result_str += "\n----------------------- Details ------------------------\n"

            for n, r in zip(
                    self.ind_to_predicates, self.result_dict[f"{mode}_{match_type}_mean_recall_collect"][100]
            ):
                result_str += "({}:{:.4f}) ".format(str(n), float(r[0]/(r[1] + 1e-5)))
            result_str += "\n--------------------------------------------------------\n\n"

        
        return result_str

    def calculate_recall(self, global_container, local_container, mode):
        # start from the 1
        for topk in topk_range:
            pred_rels_cls = torch.from_numpy(local_container['rel_cls'][:topk])
            pred_rel_vec = torch.from_numpy(local_container['rel_vec'][:topk])
            
            gt_rels = torch.from_numpy(local_container["gt_rels"])
            gt_boxes = torch.from_numpy(local_container["gt_boxes"])

            gt_box_cnter = torch.stack(((gt_boxes[:, 0] + gt_boxes[:, 2]) / 2, (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2 ), dim=1)
            gt_rel_vector = torch.cat((gt_box_cnter[gt_rels[:, 0]], gt_box_cnter[gt_rels[:, 1]]),dim=1)

            error_ratio = 0.1
            gt_rel_vec_dist_thres = (torch.abs(gt_rel_vector[:, 0] - gt_rel_vector[:,2]) 
                                                    + torch.abs(gt_rel_vector[:, 1] - gt_rel_vector[:,3])) * 0.5 * error_ratio
            

            rel_vec_dist = torch.cdist(gt_rel_vector, pred_rel_vec, p=1) / 4
            match_idx = rel_vec_dist <= gt_rel_vec_dist_thres.unsqueeze(-1) # num_gt, num_pred

            # print(gt_rel_vec_dist_thres)
            # print(rel_vec_dist.sort()[0][0, :3])

            loc_match_gt_idx = []
            det_match_gt_idx = []

            for gt_id in range(match_idx.shape[0]):
                loc_rel_match_idx = squeeze_tensor(torch.nonzero(match_idx[gt_id]))
                if len(loc_rel_match_idx) > 0 :
                    loc_match_gt_idx.append(gt_id)
                else:
                    continue
                rel_pred_cls = pred_rels_cls[loc_rel_match_idx]
                det_rel_match_idx = squeeze_tensor(torch.nonzero(rel_pred_cls == gt_rels[gt_id, -1] ))

                if len(det_rel_match_idx) > 0 :
                    det_match_gt_idx.append(gt_id)

            self.result_dict[f"{mode}_{self.type}_loc"][topk].append(len(loc_match_gt_idx) / len(gt_rels) )
            self.result_dict[f"{mode}_{self.type}_det"][topk].append(len(det_match_gt_idx) / len(gt_rels) )

            def stat_per_class_recall_hit(hit_type, gt_hit_idx):
                gt_rel_labels = gt_rels[:, -1]
                hit_rel_class_id = gt_rel_labels[gt_hit_idx]
                per_cls_rel_hit = torch.zeros(
                    (self.num_rel_cls, 2), dtype=torch.int64
                )
                # first one is pred hit num, second is gt num
                per_cls_rel_hit[hit_rel_class_id, 0] += 1
                per_cls_rel_hit[gt_rel_labels, 1] += 1
                self.result_dict[f"{mode}_{hit_type}_mean_recall_collect"][topk] += per_cls_rel_hit

            stat_per_class_recall_hit('loc', loc_match_gt_idx)
            stat_per_class_recall_hit('det', det_match_gt_idx)

class SGStagewiseRecall(SceneGraphEvaluation):
    def __init__(
            self,
            cfg,
            predicates_categories,
            result_dict,
    ):
        super(SGStagewiseRecall, self).__init__(result_dict)
        self.type = "stage_recall"
        self.cfg = cfg
        self.predicates_categories = predicates_categories
        self.num_rel_cls = len(self.predicates_categories)
        try:
            self.ent_num_classes = self.cfg.MODEL.DYHEAD.NUM_CLASSES - 1  # 150
        except:
            pass
        
        
        # the recall statistic for each categories
        # for the following visualization
        self.per_img_rel_cls_recall = []
        for _ in range(len(topk_range)):
            self.per_img_rel_cls_recall.append(
                {
                    "pair_loc": [],
                    "pair_det": [],
                    "rel_hit": [],
                    "pred_cls": [],
                }
            )

        self.relation_per_cls_hit_recall = {
            "rel_hit": torch.zeros(
                (len(topk_range), self.num_rel_cls, 2), dtype=torch.int64
            ),
            "pair_loc": torch.zeros(
                (len(topk_range), self.num_rel_cls, 2), dtype=torch.int64
            ),
            "pair_det": torch.zeros(
                (len(topk_range), self.num_rel_cls, 2), dtype=torch.int64
            ),
            "pred_cls": torch.zeros(
                (len(topk_range), self.num_rel_cls, 2), dtype=torch.int64
            ),
        }

        base_triplet_labels_all = torch.load("/public/home/v-liutao/vln/VS3_CVPR23/base_triplet_labels_all.pt")
        base_pair_labels_all = torch.load("/public/home/v-liutao/vln/VS3_CVPR23/base_pair_labels_all.pt")
        self.base_pair_labels_all = base_pair_labels_all
        self.base_triplet_labels_all = base_triplet_labels_all
        self.relation_per_cls_hit_recall_pair = {
            "rel_hit_seen_pair": torch.zeros(
                (len(topk_range), self.num_rel_cls, 2), dtype=torch.int64
            ),
            "rel_hit_unseen_pair": torch.zeros(
                (len(topk_range), self.num_rel_cls, 2), dtype=torch.int64
            ),
            "pair_loc_seen_pair": torch.zeros(
                (len(topk_range), self.num_rel_cls, 2), dtype=torch.int64
            ),
            "pair_loc_unseen_pair": torch.zeros(
                (len(topk_range), self.num_rel_cls, 2), dtype=torch.int64
            ),
            "pair_det_seen_pair": torch.zeros(
                (len(topk_range), self.num_rel_cls, 2), dtype=torch.int64
            ),
            "pair_det_unseen_pair": torch.zeros(
                (len(topk_range), self.num_rel_cls, 2), dtype=torch.int64
            ),
            "pred_cls_seen_pair": torch.zeros(
                (len(topk_range), self.num_rel_cls, 2), dtype=torch.int64
            ),
            "pred_cls_unseen_pair": torch.zeros(
                (len(topk_range), self.num_rel_cls, 2), dtype=torch.int64
            ),
        }

        self.rel_hit_types = [
            "pair_loc",
            "pair_det",
            "pred_cls",
            "rel_hit",
        ]

        self.eval_dynamic_anchor = False
        # if self.cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.DYNAMIC_REL_ANCHORS and not self.cfg.MODEL.REL_DETR.ENTITIES_AWARE_HEAD.USE_ENTITIES_PRED:
        #     self.eval_dynamic_anchor = True

        self.eval_rel_pair_prop = False
        self.rel_pn_on = False
        self.mp_pair_refine_iter = 1

        try:
            if cfg.MODEL.ROI_RELATION_HEAD.FEATURE_NECK.NAME == 'bgnn':
                self.rel_pn_on = True
                self.filter_the_mp_instance = True
                self.mp_pair_refine_iter = cfg.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.ITERATE_MP_PAIR_REFINE
        except AttributeError:
            pass

        self.vaild_rel_prop_num = 300

        if not self.rel_pn_on:
            self.filter_the_mp_instance = False

        # category clustering for overlapping
        self.instance_class_clustering = False
        self.predicate_class_clustering = False

    def register_container(self, mode):
        # the recall value for each images

        self.result_dict[f"{mode}_{self.type}_pair_loc"] = { k: [] for k in topk_range} 
        self.result_dict[f"{mode}_{self.type}_pair_det"] = { k: [] for k in topk_range} 
        self.result_dict[f"{mode}_{self.type}_rel_hit"] = { k: [] for k in topk_range} 
        self.result_dict[f"{mode}_{self.type}_pred_cls"] = { k: [] for k in topk_range} 

        self.result_dict[f"{mode}_{self.type}_box_size_Recall"] ={ k: {0: [], 1: [], 2: []} for k in topk_range} 

        self.result_dict[f"{mode}_{self.type}_rel_prop_pair_loc_before_relrpn"] = []
        self.result_dict[f"{mode}_{self.type}_rel_prop_pair_det_before_relrpn"] = []
        self.result_dict[f"{mode}_{self.type}_rel_prop_pair_loc_after_relrpn"] = []
        self.result_dict[f"{mode}_{self.type}_rel_prop_pair_det_after_relrpn"] = []

        self.result_dict[f"{mode}_{self.type}_anchor_pair_loc_hit"] = defaultdict(list)
        self.result_dict[f"{mode}_{self.type}_anchor_pair_det_hit"] = defaultdict(list)

        for i in range(self.mp_pair_refine_iter):
            self.result_dict[
                f"{mode}_{self.type}_rel_confidence_ap-iter{i}-top{self.vaild_rel_prop_num}"
            ] = []
            self.result_dict[f"{mode}_{self.type}_rel_confidence_ap-iter{i}-top100"] = []

            self.result_dict[
                f"{mode}_{self.type}_rel_confidence_auc-iter{i}-top{self.vaild_rel_prop_num}"
            ] = []
            self.result_dict[f"{mode}_{self.type}_rel_confidence_auc-iter{i}-top100"] = []

        self.result_dict[f"{mode}_{self.type}_pred_cls_auc-top100"] = []
        self.result_dict[f"{mode}_{self.type}_effective_union_pairs_rate"] = []
        self.result_dict[f"{mode}_{self.type}_effective_union_pairs_range"] = []
        self.result_dict[f"{mode}_instances_det_recall"] = []
        self.result_dict[f"{mode}_instances_loc_recall"] = []

        # todo add per cls evaluation

    def generate_res_dict(self, mode):
        res_dict = {}
        result_str = "SGG Stagewise Recall: \n"
        for each_rel_hit_type in self.rel_hit_types:
            result_str += "    "
            iter_obj = self.result_dict[f"{mode}_{self.type}_{each_rel_hit_type}"].items()
            for k, v in iter_obj:
                res_dict[f'{mode}_{each_rel_hit_type}/top{k}'] = np.mean(v)

        res_dict[f'{mode}_instances_loc_recall'] = np.mean(self.result_dict[f'{mode}_instances_loc_recall'])
        res_dict[f'{mode}_instances_det_recall'] = np.mean(self.result_dict[f'{mode}_instances_det_recall'])

        if self.eval_rel_pair_prop:

            res_dict[f'{mode}_{self.type}_effective_union_pairs_rate'] = np.mean(
                self.result_dict[f'{mode}_{self.type}_effective_union_pairs_rate'])

            res_dict[f'{mode}_{self.type}_effective_union_pairs_range'] = np.mean(
                self.result_dict[f'{mode}_{self.type}_effective_union_pairs_range'])

            res_dict[f'{mode}_{self.type}_effective_union_pairs_range'] = int(
                np.percentile(self.result_dict[f'{mode}_{self.type}_effective_union_pairs_range'], 85))

            for i in range(self.mp_pair_refine_iter):
                if len(self.result_dict[f"{mode}_{self.type}_rel_confidence_auc-iter{i}-top100"]) > 0:
                    res_dict[f'{mode}_{self.type}_rel_confidence_auc-iter{i}-top100'] = np.mean(
                        self.result_dict[f'{mode}_{self.type}_rel_confidence_auc-iter{i}-top100'])
                if len(self.result_dict[f"{mode}_{self.type}_rel_confidence_ap-iter{i}-top100"]) > 0:
                    res_dict[f'{mode}_{self.type}_rel_confidence_ap-iter{i}-top100'] = np.mean(
                        self.result_dict[f'{mode}_{self.type}_rel_confidence_ap-iter{i}-top100'])

                type_name = f"{mode}_{self.type}_rel_confidence_auc-iter{i}-top{self.vaild_rel_prop_num}"
                if (len(self.result_dict[type_name]) > 0):
                    res_dict[type_name] = np.mean(self.result_dict[type_name])

                type_name = f'{mode}_{self.type}_rel_confidence_ap-iter{i}-top{self.vaild_rel_prop_num}'
                if (len(self.result_dict[type_name]) > 0):
                    res_dict[type_name] = np.mean(self.result_dict[type_name])

        type_name = f"{mode}_{self.type}_pred_cls_auc-top100"
        if len(self.result_dict[type_name]) > 0:
            res_dict[type_name] = np.mean(self.result_dict[type_name])

        return res_dict

    def generate_print_string(self, mode):
        result_str = "SGG Stagewise Recall: \n"
        for each_rel_hit_type in self.rel_hit_types:
            result_str += "    "
            if isinstance(self.result_dict[f"{mode}_{self.type}_{each_rel_hit_type}"], dict):
                iter_obj = self.result_dict[f"{mode}_{self.type}_{each_rel_hit_type}"].items()
            else:
                iter_obj = [
                    (4096, vals)
                    for vals in self.result_dict[f"{mode}_{self.type}_{each_rel_hit_type}"]
                ]
            for k, v in iter_obj:
                result_str += " R @ %d: %.4f; " % (k, float(np.mean(v)))
            result_str += f" for mode={mode}, type={each_rel_hit_type}"
            result_str += "\n"
        result_str += "\n"

        result_str += (
            "instances detection recall:\n"
            f"locating: {np.mean(self.result_dict[f'{mode}_instances_loc_recall']):.4f}\n"
            f"detection: {np.mean(self.result_dict[f'{mode}_instances_det_recall']):.4f}\n"
        )
        result_str += "\n"

        if self.eval_rel_pair_prop:
            result_str += "effective relationship union pairs statistics \n"
            result_str += (
                f"effective relationship union pairs_rate (avg): "
                f"{np.mean(self.result_dict[f'{mode}_{self.type}_effective_union_pairs_rate']) : .3f}\n"
            )

            result_str += (
                f"effective relationship union pairs range(avg(percentile_85)/total): "
                f"{int(np.mean(self.result_dict[f'{mode}_{self.type}_effective_union_pairs_range']) + 1)}"
                f"({int(np.percentile(self.result_dict[f'{mode}_{self.type}_effective_union_pairs_range'], 85))}) / "
                f"{self.eval_rel_pair_prop} \n\n"
            )

            for i in range(self.mp_pair_refine_iter):
                if len(self.result_dict[f"{mode}_{self.type}_rel_confidence_auc-iter{i}-top100"]) > 0:
                    result_str += (
                        f"The AUC of relpn (stage {i})-top100: "
                        f"{np.mean(self.result_dict[f'{mode}_{self.type}_rel_confidence_auc-iter{i}-top100']): .3f} \n"
                    )

                if len(self.result_dict[f"{mode}_{self.type}_rel_confidence_ap-iter{i}-top100"]) > 0:
                    result_str += (
                        f"The AP of relpn (stage {i})-top100: "
                        f"{np.mean(self.result_dict[f'{mode}_{self.type}_rel_confidence_ap-iter{i}-top100']): .3f} \n"
                    )

                if (
                        len(
                            self.result_dict[
                                f"{mode}_{self.type}_rel_confidence_auc-iter{i}-top{self.vaild_rel_prop_num}"
                            ]
                        )
                        > 0
                ):
                    result_str += (
                        f"The AUC of relpn (stage {i})-top{self.vaild_rel_prop_num}: "
                        f"{np.mean(self.result_dict[f'{mode}_{self.type}_rel_confidence_auc-iter{i}-top{self.vaild_rel_prop_num}']): .3f} \n"
                    )

                if (
                        len(
                            self.result_dict[
                                f"{mode}_{self.type}_rel_confidence_ap-iter{i}-top{self.vaild_rel_prop_num}"
                            ]
                        )
                        > 0
                ):
                    result_str += (
                        f"The AP of relpn (stage {i})-top{self.vaild_rel_prop_num}: "
                        f"{np.mean(self.result_dict[f'{mode}_{self.type}_rel_confidence_ap-iter{i}-top{self.vaild_rel_prop_num}']): .3f} \n"
                    )

        if len(self.result_dict[f"{mode}_{self.type}_pred_cls_auc-top100"]) > 0:
            result_str += (
                f"The AUC of pred_clssifier: "
                f"{np.mean(self.result_dict[f'{mode}_{self.type}_pred_cls_auc-top100']): .3f} \n"
            )
        for topk in self.result_dict[f"{mode}_{self.type}_box_size_Recall"].keys():
            result_str += f"\nR@{topk} in different box size: \n"
            for i, box_type in enumerate(['both_small(<64x64)', 'one_small', 'both_big']):
                r_100_v = np.mean(self.result_dict[f"{mode}_{self.type}_box_size_Recall"][topk][i])
                result_str += f"{box_type}: {r_100_v:.5f}; "

        if len(self.result_dict[f"{mode}_{self.type}_anchor_pair_det_hit"]) > 0 :
            result_str += "\nDynamic anchor retrival performance:\n"
            for match_type in [f"{mode}_{self.type}_anchor_pair_det_hit", f"{mode}_{self.type}_anchor_pair_loc_hit"]:
                result_str += "   "
                for topk_num in self.result_dict[match_type].keys():
                    loc_hit_recall = np.mean(self.result_dict[match_type][topk_num])
                    result_str += f"R@{topk_num}: {loc_hit_recall:.5f};  "
                result_str += f"{match_type}\n"

        result_str += "\n"

        return result_str

    def calculate_recall(
            self,
            mode,
            global_container,
            gt_boxlist,
            gt_relations,
            gt_triplets,
            pred_boxlist,
            pred_rel_pair_idx,
            pred_rel_dist,
    ):
        """
        evaluate stage-wise recall on one images

        :param global_container:
        :param gt_boxlist: ground truth BoxList
        :param gt_relations: ground truth relationships: np.array (subj_instance_id, obj_instance_id, rel_cate_id)
        :param pred_boxlist: prediction  BoxList
         the rel predictions has already been sorted in descending.
        :param pred_rel_pair_idx: prediction relationship instances pairs index  np.array (n, 2)
        :param pred_rel_scores: prediction relationship predicate scores  np.array  (n, )
        :param eval_rel_pair_prop: prediction relationship instance pair proposals  Top 2048 for for top100 selection
        :return:
        """
        gt_triplets = torch.from_numpy(gt_triplets)
        # import ipdb; ipdb.set_trace()
        # store the hit index between the ground truth and predictions
        hit_idx = {"rel_hit": [], "pair_det_hit": [], "pair_loc_hit": [], "pred_cls_hit": []}

        if self.eval_rel_pair_prop:
            hit_idx["prop_pair_det_hit"] = []
            hit_idx["prop_pair_loc_hit"] = []

        device = torch.zeros((1, 1)).cpu().device  # cpu_device

        iou_thres = global_container["iou_thres"]

        # transform every array to tensor for adapt the previous code
        # (num_rel, 3) = subj_id, obj_id, rel_labels

        pred_rel_dist = pred_boxlist.get_field("pred_rel_scores")[:, :]
        rel_scores = torch.max(pred_boxlist.get_field("pred_rel_scores")[:, :], dim=-1)[0]
        pred_rels_cls = pred_boxlist.get_field("pred_rel_labels") - 1
        pred_rels = torch.from_numpy(
            np.column_stack((pred_rel_pair_idx, pred_rels_cls))
        )
        # (num_rel, )

        instance_hit_iou = boxlist_iou(pred_boxlist, gt_boxlist, to_cuda=False)

        box_size_img = []
        for each in gt_boxlist.area():
            box_size_img.append(each ** 0.5)

        rel_ent_box_marker = []
        for each_rel in gt_relations.numpy():
            sub_id = each_rel[0]
            obj_id = each_rel[1]
            box_pair_area = torch.Tensor([box_size_img[sub_id], box_size_img[obj_id]])
            big_box_num = torch.sum(box_pair_area > 64).item()
            rel_ent_box_marker.append(big_box_num)

        instance_hit_iou = instance_hit_iou.to(device)
        if len(instance_hit_iou) == 0:
            # todo add zero to final results
            pass

        # box pair location hit
        # check the locate results
        inst_loc_hit_idx = instance_hit_iou >= iou_thres
        # (N, 2) array, indicate the which pred box idx matched which gt box idx
        inst_loc_hit_idx = torch.nonzero(inst_loc_hit_idx)
        pred_box_loc_hit_idx = inst_loc_hit_idx[:, 0]
        gt_box_loc_hit_idx = inst_loc_hit_idx[:, 1]

        # store the pred box idx hit gt box idx set:
        # the box prediction and gt box are N to M relation,
        # which means one box prediction may hit multiple gt box,
        # so we need to store the each pred box hit gt boxes in set()
        loc_box_matching_results = defaultdict(set)  # key: pred-box index, val: gt-box index
        for each in inst_loc_hit_idx:
            loc_box_matching_results[each[0].item()].add(each[1].item())

        # base on the location results, check the classification results
        gt_det_label_to_cmp = gt_boxlist.get_field("labels")[gt_box_loc_hit_idx]
        pred_det_label_to_cmp = pred_boxlist.get_field("pred_labels")[pred_box_loc_hit_idx]

        # todo working on category clustering later
        if self.instance_class_clustering:
            gt_det_label_to_cmp = copy.deepcopy(gt_det_label_to_cmp)
            pred_det_label_to_cmp = copy.deepcopy(pred_det_label_to_cmp)
            pred_det_label_to_cmp, gt_det_label_to_cmp = trans_cluster_label(
                pred_det_label_to_cmp, gt_det_label_to_cmp, ENTITY_CLUSTER
            )

        pred_det_hit_stat = pred_det_label_to_cmp == gt_det_label_to_cmp

        pred_box_det_hit_idx = pred_box_loc_hit_idx[pred_det_hit_stat]
        gt_box_det_hit_idx = gt_box_loc_hit_idx[pred_det_hit_stat]

        self.result_dict[f"{mode}_instances_det_recall"].append(
            len(torch.unique(gt_box_det_hit_idx)) / (len(gt_boxlist) + 0.000001)
        )
        self.result_dict[f"{mode}_instances_loc_recall"].append(
            len(torch.unique(gt_box_loc_hit_idx)) / (len(gt_boxlist) + 0.000001)
        )
        # store the detection results in matching dict
        det_box_matching_results = defaultdict(set)
        for idx in range(len(pred_box_det_hit_idx)):
            det_box_matching_results[pred_box_det_hit_idx[idx].item()].add(
                gt_box_det_hit_idx[idx].item()
            )

        # after the entities detection recall check, then the entities pairs locating classifications check
        def get_entities_pair_locating_n_cls_hit(to_cmp_pair_mat):
            # according to the detection box hit results,
            # check the location and classification hit of entities pairs
            # instances box location hit res
            rel_loc_pair_mat, rel_loc_init_pred_idx = dump_hit_indx_dict_to_tensor(
                to_cmp_pair_mat, loc_box_matching_results
            )
            # instances box location and category hit
            rel_det_pair_mat, rel_det_init_pred_idx = dump_hit_indx_dict_to_tensor(
                to_cmp_pair_mat, det_box_matching_results
            )
            rel_pair_mat = copy.deepcopy(rel_det_pair_mat)
            rel_init_pred_idx = copy.deepcopy(rel_det_init_pred_idx)

            # use the intersect operate to calculate how the prediction relation pair hit the gt relationship
            # pairs,
            # first is the box pairs location hit and detection hit separately
            rel_loc_hit_idx = (
                intersect_2d_torch_tensor(rel_loc_pair_mat, gt_relations[:, :2])
                    .nonzero()
                    .transpose(1, 0)
            )
            # the index of prediction box hit the ground truth
            pred_rel_loc_hit_idx = rel_loc_init_pred_idx[rel_loc_hit_idx[0]]
            gt_rel_loc_hit_idx = rel_loc_hit_idx[1]  # the prediction hit ground truth index

            rel_det_hit_idx = (
                intersect_2d_torch_tensor(rel_det_pair_mat, gt_relations[:, :2])
                    .nonzero()
                    .transpose(1, 0)
            )
            pred_rel_det_hit_idx = rel_det_init_pred_idx[rel_det_hit_idx[0]]
            gt_rel_det_hit_idx = rel_det_hit_idx[1]

            return (
                rel_loc_pair_mat,
                rel_loc_init_pred_idx,
                rel_pair_mat,
                rel_init_pred_idx,
                pred_rel_loc_hit_idx,
                gt_rel_loc_hit_idx,
                pred_rel_det_hit_idx,
                gt_rel_det_hit_idx,
            )

        # check relation proposal recall
        if self.eval_rel_pair_prop:
            # before relationship rpn
            # prop_rel_pair_mat, prop_rel_init_pred_idx, \
            # prop_rel_loc_hit_idx, prop_rel_loc_hit_gt_idx, \
            # prop_rel_det_hit_idx, prop_rel_det_hit_gt_idx = get_entities_pair_locating_n_cls_hit(rel_pair_prop.pair_mat)
            # rel_proposal_pair_loc_hit_cnt_before_rpn = len(torch.unique(prop_rel_loc_hit_gt_idx))
            # rel_proposal_pair_det_hit_cnt_before_rpn = len(torch.unique(prop_rel_det_hit_gt_idx))

            # after relationship rpn
            (
                prop_rel_loc_pair_mat,
                prop_rel_loc_init_pred_idx,
                prop_rel_pair_mat,
                prop_rel_init_pred_idx,
                prop_rel_loc_hit_idx,
                prop_rel_loc_hit_gt_idx,
                prop_rel_det_hit_idx,
                prop_rel_det_hit_gt_idx,
            ) = get_entities_pair_locating_n_cls_hit(pred_rel_pair_idx)

            rel_proposal_pair_loc_hit_cnt_after_rpn = len(
                torch.unique(prop_rel_loc_hit_gt_idx)
            )
            rel_proposal_pair_det_hit_cnt_after_rpn = len(
                torch.unique(prop_rel_det_hit_gt_idx)
            )

            # self.rel_recall_per_img[topk_idx]['rel_prop_pair_loc_before_relrpn'] \
            #     .append(rel_proposal_pair_loc_hit_cnt_before_rpn / (float(gt_relations.shape[0]) + 0.00001))
            # self, .rel_recall_per_img[topk_idx]['rel_prop_pair_det_before_relrpn'] \
            #     .append(rel_proposal_pair_det_hit_cnt_before_rpn / (float(gt_relations.shape[0]) + 0.00001))
            self.result_dict[f"{mode}_{self.type}_rel_prop_pair_loc_after_relrpn"].append(
                rel_proposal_pair_loc_hit_cnt_after_rpn
                / (float(gt_relations.shape[0]) + 0.00001)
            )
            self.result_dict[f"{mode}_{self.type}_rel_prop_pair_det_after_relrpn"].append(
                rel_proposal_pair_det_hit_cnt_after_rpn
                / (float(gt_relations.shape[0]) + 0.00001)
            )
            self.result_dict[f"{mode}_{self.type}_effective_union_pairs_rate"].append(
                len(prop_rel_loc_hit_idx) / (float(pred_rel_pair_idx.shape[0]) + 0.00001)
            )
            if len(prop_rel_loc_hit_idx) > 0:
                self.result_dict[f"{mode}_{self.type}_effective_union_pairs_range"].append(
                    np.percentile(prop_rel_loc_hit_idx, 95)
                )
            else:
                self.result_dict[f"{mode}_{self.type}_effective_union_pairs_range"].append(
                    self.eval_rel_pair_prop
                )

        # eval the relness and pred clser ranking performance for postive samples

        def eval_roc(scores, matching_results, roc_pred_range=300):
            ref_labels = torch.zeros_like(scores)
            ref_labels[matching_results] = 1
            val, sort_idx = torch.sort(scores, descending=True)
            y = ref_labels[sort_idx[:roc_pred_range]]
            pred = scores[sort_idx[:roc_pred_range]]

            # no auc when no postive samples of no negative samples
            if len(torch.nonzero(y)) > 0 and len(torch.nonzero(y)) != len(y):
                y = y.detach().long().cpu().numpy()
                pred = pred.detach().cpu().numpy()
                fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
                auc = metrics.auc(fpr, tpr)
            else:
                auc = np.nan
                thresholds = None
                fpr = None
                tpr = None

            roc_res = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": auc}
            return roc_res

        def eval_ap(pred, matched_idx, gt_idx, total_gt_num, pred_range=300):
            # tp + fn

            posb_tp = torch.ones(pred.shape[0], dtype=torch.long) * -1
            posb_tp[matched_idx] = gt_idx
            pred_score, pred_idx = torch.sort(pred, descending=True)

            pred_idx = pred_idx[:pred_range]
            pred_score = pred_score[:pred_range]

            pr_s = []
            recs = []

            for thres in range(1, 10):
                thres *= 0.1
                all_p_idx = pred_score > thres
                all_p_idx = pred_idx[all_p_idx]

                tp_idx = posb_tp >= 0
                mask = torch.zeros(tp_idx.shape[0], dtype=torch.bool)
                mask[all_p_idx] = True
                tp_idx = tp_idx & mask

                tp = len(torch.unique(posb_tp[tp_idx]))

                fp_idx = posb_tp < 0
                mask = torch.zeros(fp_idx.shape[0], dtype=torch.bool)
                mask[all_p_idx] = True
                fp_idx = fp_idx & mask

                fp = len(torch.unique(posb_tp[fp_idx]))

                pr = tp / (tp + fp + 0.0001)
                rec = tp / (total_gt_num + 0.0001)

                pr_s.append(pr)
                recs.append(rec)

            def get_ap(rec, prec):
                """Compute AP given precision and recall."""
                # correct AP calculation
                # first append sentinel values at the end
                mrec = np.concatenate(([0.0], rec, [1.0]))
                mpre = np.concatenate(([0.0], prec, [0.0]))

                # compute the precision envelope
                for i in range(mpre.size - 1, 0, -1):
                    mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

                # to calculate area under PR curve, look for points
                # where X axis (recall) changes value
                i = np.where(mrec[1:] != mrec[:-1])[0]

                # and sum (\Delta recall) * prec
                ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
                return ap

            return get_ap(np.array(recs), np.array(pr_s))

        if self.rel_pn_on:
            relness_score = pred_boxlist.get_field("rel_confidence")
            for i in range(relness_score.shape[-1]):

                # if len()
                roc_res = eval_roc(
                    relness_score[:, i], prop_rel_loc_hit_idx, self.vaild_rel_prop_num
                )

                ap_res = eval_ap(
                    relness_score[:, i],
                    prop_rel_loc_hit_idx,
                    prop_rel_loc_hit_gt_idx,
                    float(gt_relations.shape[0]),
                    self.vaild_rel_prop_num,
                )

                auc = roc_res["auc"]

                self.result_dict[
                    f"{mode}_{self.type}_rel_confidence_ap-iter{i}-top{self.vaild_rel_prop_num}"
                ].append(ap_res)

                if not np.isnan(auc):
                    self.result_dict[
                        f"{mode}_{self.type}_rel_confidence_auc-iter{i}-top{self.vaild_rel_prop_num}"
                    ].append(auc)

                roc_res = eval_roc(relness_score[:, i], prop_rel_loc_hit_idx, 100)
                ap_res = eval_ap(
                    relness_score[:, i],
                    prop_rel_loc_hit_idx,
                    prop_rel_loc_hit_gt_idx,
                    float(gt_relations.shape[0]),
                    100,
                )
                auc = roc_res["auc"]

                self.result_dict[f"{mode}_{self.type}_rel_confidence_ap-iter{i}-top100"].append(ap_res)

                if not np.isnan(auc):
                    self.result_dict[f"{mode}_{self.type}_rel_confidence_auc-iter{i}-top100"].append(
                        auc
                    )


        # for different top-K relationship filtering, check the recall
        for topk_idx, topk in enumerate(topk_range):
            selected_rel_pred = pred_rels[:topk]
            # count the detection recall
            # instance_det_hit_num[topk_idx] += len(torch.unique(gt_box_det_hit_idx))
            # instance_det_recall_per_img[topk_idx] \
            #     .append(len(torch.unique(gt_box_det_hit_idx)) / (len(gt_boxes)))

            # after collect the pred box hit result,
            # now need to check the hit of each triplets in gt rel set
            (
                rel_loc_pair_mat,
                rel_loc_init_pred_idx,
                rel_pair_mat,
                rel_init_pred_idx,
                pred_rel_loc_hit_idx,
                gt_rel_loc_hit_idx,
                pred_rel_det_hit_idx,
                gt_rel_det_hit_idx,
            ) = get_entities_pair_locating_n_cls_hit(selected_rel_pred[:, :2])

            if topk == 100:
                pred_rel_dist = pred_boxlist.get_field("pred_rel_scores")[:, 1:] # (#pred_rels, num_pred_class)
                rel_scores = torch.max(pred_boxlist.get_field("pred_rel_scores")[:, 1:], dim=-1)[0] # (#pred_rels, ), 最终预测label的置信度
                rel_class = pred_boxlist.get_field("pred_rel_labels") - 1 # (#pred_rels, ), 最终预测label的标签
                # rel_scores, rel_class = pred_rel_dist[:, 1:].max(dim=1)
                det_score = pred_boxlist.get_field("pred_scores")
                pairs = pred_boxlist.get_field("rel_pair_idxs").long()

                rel_scores_condi_det = (
                        rel_scores * det_score[pairs[:, 0]] * det_score[pairs[:, 1]]
                )
                rel_scores_condi_det = rel_scores_condi_det[:topk]

                if not torch.isnan(rel_scores_condi_det).any():
                    roc_res = eval_roc(rel_scores_condi_det, pred_rel_loc_hit_idx, topk)
                    if not np.isnan(roc_res["auc"]):
                        self.result_dict[f"{mode}_{self.type}_pred_cls_auc-top{topk}"].append(
                            roc_res["auc"]
                        )
            

            def predicates_category_clustering(pred_labels):
                gt_pred_labels = copy.deepcopy(gt_relations[:, -1])
                rel_predicate_label, gt_pred_labels = trans_cluster_label(
                    pred_labels, gt_pred_labels, PREDICATE_CLUSTER
                )
                to_cmp_gt_relationships = copy.deepcopy(gt_relations)
                to_cmp_gt_relationships[:, -1] = gt_pred_labels
                return rel_predicate_label, to_cmp_gt_relationships

            # stage-wise evaluation
            # then we evaluate the full relationship triplets, sub obj detection and predicates
            rel_predicate_label = copy.deepcopy(selected_rel_pred[:, -1][rel_init_pred_idx])
            rel_loc_pair_pred_label = copy.deepcopy(
                selected_rel_pred[:, -1][rel_loc_init_pred_idx]
            )

            to_cmp_gt_relationships = gt_relations
            if self.predicate_class_clustering:
                (
                    rel_loc_pair_pred_label,
                    to_cmp_gt_relationships,
                ) = predicates_category_clustering(rel_loc_pair_pred_label)
                rel_predicate_label, to_cmp_gt_relationships = predicates_category_clustering(
                    rel_predicate_label
                )

            rel_predicate_label.unsqueeze_(1)

            # eval relationship detection (entities + predicates)
            rel_pair_mat = torch.cat((rel_pair_mat, rel_predicate_label), dim=1)
            rel_hit_idx = (
                intersect_2d_torch_tensor(rel_pair_mat, to_cmp_gt_relationships)
                    .nonzero()
                    .transpose(1, 0)
            )
            pred_rel_hit_idx = rel_init_pred_idx[rel_hit_idx[0]]
            gt_rel_hit_idx = rel_hit_idx[1]

            # eval relationship predicate classification (entities pair loc + predicates)

            rel_loc_pair_pred_label.unsqueeze_(1)
            pred_cls_matrix = torch.cat((rel_loc_pair_mat, rel_loc_pair_pred_label), dim=1)
            pred_cls_hit_idx = (
                intersect_2d_torch_tensor(pred_cls_matrix, to_cmp_gt_relationships)
                    .nonzero()
                    .transpose(1, 0)
            )
            pred_predicate_cls_hit_idx = rel_loc_init_pred_idx[pred_cls_hit_idx[0]]
            gt_pred_cls_hit_idx = pred_cls_hit_idx[1]

            # statistic the prediction results
            # per-class recall
            def stat_per_class_recall_hit(self, hit_type, gt_hit_idx):
                gt_rel_labels = gt_relations[:, -1]
                hit_rel_class_id = gt_rel_labels[gt_hit_idx]
                per_cls_rel_hit = torch.zeros(
                    (self.num_rel_cls, 2), dtype=torch.int64
                )
                # first one is pred hit num, second is gt num
                per_cls_rel_hit[hit_rel_class_id, 0] += 1
                per_cls_rel_hit[gt_rel_labels, 1] += 1
                self.relation_per_cls_hit_recall[hit_type][topk_idx] += per_cls_rel_hit
                self.per_img_rel_cls_recall[topk_idx][hit_type].append(per_cls_rel_hit)
            
            def stat_per_class_recall_hit_pair(self, hit_type, gt_hit_idx):
                gt_rel_labels = gt_triplets[:, -1]
                hit_rel_class_id = gt_rel_labels[gt_hit_idx]
                per_cls_rel_hit_seen_pair = torch.zeros(
                    (self.num_rel_cls, 2), dtype=torch.int64
                )
                per_cls_rel_hit_unseen_pair = torch.zeros(
                    (self.num_rel_cls, 2), dtype=torch.int64
                )
                gt_rel_pairs = gt_triplets[gt_hit_idx]
                for pair in gt_rel_pairs:
                    sub, obj, pred = pair[0].item(), pair[1].item(), pair[2].item()
                    if (sub, obj) in self.base_pair_labels_all and per_cls_rel_hit_seen_pair[pred, 0] == 0:
                        per_cls_rel_hit_seen_pair[pred, 0] += 1
                    elif (sub, obj) not in self.base_pair_labels_all and per_cls_rel_hit_unseen_pair[pred, 0] == 0:
                        per_cls_rel_hit_unseen_pair[pred, 0] += 1
                for pair in gt_triplets:
                    sub, obj, pred = pair[0].item(), pair[1].item(), pair[2].item()
                    if (sub, obj) in self.base_pair_labels_all and per_cls_rel_hit_seen_pair[pred, 1] == 0:
                        per_cls_rel_hit_seen_pair[pred, 1] += 1
                    elif (sub, obj) not in self.base_pair_labels_all and per_cls_rel_hit_unseen_pair[pred, 1] == 0:
                        per_cls_rel_hit_unseen_pair[pred, 1] += 1
                # first one is pred hit num, second is gt num
                self.relation_per_cls_hit_recall_pair[hit_type+'_seen_pair'][topk_idx] += per_cls_rel_hit_seen_pair
                self.relation_per_cls_hit_recall_pair[hit_type+'_unseen_pair'][topk_idx] += per_cls_rel_hit_unseen_pair

            stat_per_class_recall_hit(self, "rel_hit", gt_rel_hit_idx)
            stat_per_class_recall_hit(self, "pair_loc", gt_rel_loc_hit_idx)
            stat_per_class_recall_hit(self, "pair_det", gt_rel_det_hit_idx)
            stat_per_class_recall_hit(self, "pred_cls", gt_pred_cls_hit_idx)

            stat_per_class_recall_hit_pair(self, "rel_hit", gt_rel_hit_idx)
            stat_per_class_recall_hit_pair(self, "pair_loc", gt_rel_loc_hit_idx)
            stat_per_class_recall_hit_pair(self, "pair_det", gt_rel_det_hit_idx)
            stat_per_class_recall_hit_pair(self, "pred_cls", gt_pred_cls_hit_idx)

            # pre image relationship pairs hit counting
            rel_hit_cnt = len(torch.unique(gt_rel_hit_idx))
            pair_det_hit_cnt = len(torch.unique(gt_rel_det_hit_idx))
            pred_cls_hit_cnt = len(torch.unique(gt_pred_cls_hit_idx))
            pair_loc_hit_cnt = len(torch.unique(gt_rel_loc_hit_idx))

            self.result_dict[f"{mode}_{self.type}_pair_loc"][topk].append(
                pair_loc_hit_cnt / (float(gt_relations.shape[0]) + 0.00001)
            )
            self.result_dict[f"{mode}_{self.type}_pair_det"][topk].append(
                pair_det_hit_cnt / (float(gt_relations.shape[0]) + 0.00001)
            )
            self.result_dict[f"{mode}_{self.type}_rel_hit"][topk].append(
                rel_hit_cnt / (float(gt_relations.shape[0]) + 0.00001)
            )
            self.result_dict[f"{mode}_{self.type}_pred_cls"][topk].append(
                pred_cls_hit_cnt / (float(gt_relations.shape[0]) + 0.00001)
            )
            # box size eval
            rel_gt_num_box_size_all = {0: 0, 1: 0, 2: 0}
            for gt_idx in rel_ent_box_marker:
                rel_gt_num_box_size_all[gt_idx] += 1
            rel_gt_num_box_size_match = {0: 0, 1: 0, 2: 0}
            for gt_idx in torch.unique(gt_rel_hit_idx).numpy():
                rel_gt_num_box_size_match[rel_ent_box_marker[gt_idx]] += 1

            if topk in self.result_dict[f"{mode}_{self.type}_box_size_Recall"].keys():
                for box_ in range(3):
                    if rel_gt_num_box_size_all[box_] != 0:
                        self.result_dict[f"{mode}_{self.type}_box_size_Recall"][topk][box_].append(
                            rel_gt_num_box_size_match[box_] / rel_gt_num_box_size_all[box_]
                        )



        if self.eval_dynamic_anchor:
            for topk_idx, topk in enumerate(topk_range):
                loc_box_matching_results_collect = {}
                det_box_matching_results_collect = {}
                for rolename in ['sub', 'obj']:
                    # check the prediction box quality
                    pred_boxes = pred_boxlist.get_field(f"dyna_anchor_{rolename}_box").clone().cpu()[:topk]
                    pred_labels = pred_boxlist.get_field(f"dyna_anchor_{rolename}_cls")[:, :self.ent_num_classes].max(-1)[1].cpu()[:topk]

                    tgt_boxes = gt_boxlist.bbox.cpu()
                    tgt_labels = gt_boxlist.get_field("labels").cpu()

                    box_giou = generalized_box_iou(pred_boxes, tgt_boxes).detach()
                    box_match_idx = box_giou >= 0.5
                    inst_loc_hit_idx = torch.nonzero(box_match_idx)
                    pred_box_loc_hit_idx = inst_loc_hit_idx[:, 0]
                    gt_box_loc_hit_idx = inst_loc_hit_idx[:, 1]

                    loc_box_matching_results = defaultdict(set)  # key: pred-box index, val: gt-box index
                    for each in inst_loc_hit_idx:
                        loc_box_matching_results[each[0].item()].add(each[1].item())
                    loc_box_matching_results_collect[rolename] = loc_box_matching_results

                    gt_det_label_to_cmp = pred_labels[pred_box_loc_hit_idx]
                    pred_det_label_to_cmp = tgt_labels[gt_box_loc_hit_idx]

                    pred_det_hit_stat = pred_det_label_to_cmp == gt_det_label_to_cmp

                    pred_box_det_hit_idx = pred_box_loc_hit_idx[pred_det_hit_stat]
                    gt_box_det_hit_idx = gt_box_loc_hit_idx[pred_det_hit_stat]

                    det_box_matching_results = defaultdict(set) # key: pred-box index, val: gt-box index
                    for idx in range(len(pred_box_det_hit_idx)): 
                        det_box_matching_results[pred_box_det_hit_idx[idx].item()].add(
                            gt_box_det_hit_idx[idx].item()
                        )
                    det_box_matching_results_collect[rolename] = det_box_matching_results
                
                def generate_pair_match_idx(gt_box_hit_idx_dict):
                    to_cmp_pair_mat = []
                    initial_pred_idx_seg = []
                    # write result into the pair mat
                    for pred_idx in range(len(pred_boxes)):
                        sub_pred_hit_idx_set = gt_box_hit_idx_dict['sub'][pred_idx]
                        obj_pred_hit_idx_set = gt_box_hit_idx_dict['obj'][pred_idx]
                        # expand the prediction index by full combination
                        for each_sub_hit_idx in sub_pred_hit_idx_set:
                            for each_obj_hit_idx in obj_pred_hit_idx_set:
                                to_cmp_pair_mat.append([each_sub_hit_idx, each_obj_hit_idx])
                                initial_pred_idx_seg.append(pred_idx)  #
                    if len(to_cmp_pair_mat) == 0:
                        to_cmp_pair_mat = torch.zeros((0, 2), dtype=torch.int64)
                    else:
                        to_cmp_pair_mat = torch.from_numpy(np.array(to_cmp_pair_mat, dtype=np.int64))

                    initial_pred_idx_seg = torch.from_numpy(np.array(initial_pred_idx_seg, dtype=np.int64))
                    return to_cmp_pair_mat, initial_pred_idx_seg

                rel_loc_pair_mat, rel_loc_init_pred_idx = generate_pair_match_idx(
                    loc_box_matching_results_collect
                )
                # instances box location and category hit
                rel_det_pair_mat, rel_det_init_pred_idx = generate_pair_match_idx(
                    det_box_matching_results_collect
                )

                # use the intersect operate to calculate how the prediction relation pair hit the gt relationship
                # pairs,
                # first is the box pairs location hit and detection hit separately
                gt_relation_pair_idx = gt_boxlist.get_field('relation_tuple').cpu()[:, :2]
                rel_loc_hit_idx = (
                    intersect_2d_torch_tensor(rel_loc_pair_mat, gt_relation_pair_idx)
                        .nonzero()
                        .transpose(1, 0)
                )
                # the index of prediction box hit the ground truth
                pred_rel_loc_hit_idx = rel_loc_init_pred_idx[rel_loc_hit_idx[0]]
                gt_rel_loc_hit_idx = rel_loc_hit_idx[1]  # the prediction hit ground truth index

                rel_det_hit_idx = (
                    intersect_2d_torch_tensor(rel_det_pair_mat, gt_relation_pair_idx)
                        .nonzero()
                        .transpose(1, 0)
                )
                pred_rel_det_hit_idx = rel_det_init_pred_idx[rel_det_hit_idx[0]]
                gt_rel_det_hit_idx = rel_det_hit_idx[1]

                loc_pair_hit = len(torch.unique(gt_rel_loc_hit_idx)) / (len(gt_relation_pair_idx) + 1e-3)
                det_pair_hit = len(torch.unique(gt_rel_det_hit_idx)) / (len(gt_relation_pair_idx) + 1e-3)

                self.result_dict[f"{mode}_{self.type}_anchor_pair_loc_hit"][topk].append(loc_pair_hit)
                self.result_dict[f"{mode}_{self.type}_anchor_pair_det_hit"][topk].append(det_pair_hit)

def squeeze_tensor(tensor):
    tensor = torch.squeeze(tensor)
    try:
        len(tensor)
    except TypeError:
        tensor.unsqueeze_(0)
    return tensor

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    # vallina box iou
    # modified from torchvision to also return the union
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    iou = inter / union

    # iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def boxlist_iou(boxlist1, boxlist2, to_cuda=True):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
            "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    N = len(boxlist1)
    M = len(boxlist2)

    if to_cuda:
        if boxlist1.bbox.device.type != 'cuda':
            boxlist1.bbox = boxlist1.bbox.cuda()
        if boxlist2.bbox.device.type != 'cuda':
            boxlist2.bbox = boxlist2.bbox.cuda()

    box1 = boxlist1.bbox
    box2 = boxlist2.bbox

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def intersect_2d_torch_tensor(x1, x2):
    return torch.from_numpy(intersect_2d(x1.numpy(), x2.numpy()))


def dump_hit_indx_dict_to_tensor(pred_pair_mat, gt_box_hit_idx_dict):
    """
    for compare the prediction and gt easily, we need to expand the N to M box match results to
    array.
    here, give relationship prediction pair matrix, expand the gt_box_hit_idx_dit to the array.
    We do the full connection of hit gt box idx of each prediction pairs
    :param pred_pair_mat:
    :param gt_box_hit_idx_dict: the hit gt idx of each prediction box
    :return:
        to_cmp_pair_mat: expanded relationship pair result (N, 2), store the gt box indexs.
            N is large than initial prediction pair matrix
        initial_pred_idx_seg: marking the seg for each pred pairs. If it hit multiple detection gt,
            it could have more than one prediction pairs, we need to mark that they are indicated to
            same initial predations
    """
    to_cmp_pair_mat = []
    initial_pred_idx_seg = []
    # write result into the pair mat
    for pred_idx, pred_pair in enumerate(pred_pair_mat):
        sub_pred_hit_idx_set = gt_box_hit_idx_dict[pred_pair[0].item()]
        obj_pred_hit_idx_set = gt_box_hit_idx_dict[pred_pair[1].item()]
        # expand the prediction index by full combination
        for each_sub_hit_idx in sub_pred_hit_idx_set:
            for each_obj_hit_idx in obj_pred_hit_idx_set:
                to_cmp_pair_mat.append([each_sub_hit_idx, each_obj_hit_idx])
                initial_pred_idx_seg.append(pred_idx)  #
    if len(to_cmp_pair_mat) == 0:
        to_cmp_pair_mat = torch.zeros((0, 2), dtype=torch.int64)
    else:
        to_cmp_pair_mat = torch.from_numpy(np.array(to_cmp_pair_mat, dtype=np.int64))

    initial_pred_idx_seg = torch.from_numpy(np.array(initial_pred_idx_seg, dtype=np.int64))
    return to_cmp_pair_mat, initial_pred_idx_seg


LONGTAIL_CATE_IDS_DICT = {
    'head': [31, 20, 22, 30, 48],
    'body': [29, 50, 1, 21, 8, 43, 40, 49, 41, 23, 7, 6, 19, 33, 16, 38],
    'tail': [11, 14, 46, 37, 13, 24, 4, 47, 5, 10, 9, 34, 3, 25, 17, 35, 42, 27, 12, 28,
             39, 36, 2, 15, 44, 32, 26, 18, 45]
}

LONGTAIL_CATE_IDS_QUERY = {}
for long_name, cate_id in LONGTAIL_CATE_IDS_DICT.items():
    for each_cate_id in cate_id:
        LONGTAIL_CATE_IDS_QUERY[each_cate_id] = long_name

PREDICATE_CLUSTER = [[50, 20, 9], [22, 48, 49], [31], [31, 41, 1], [31, 30]]
ENTITY_CLUSTER = [[91, 149, 53, 78, 20, 79, 90, 56, 68]]


def get_cluster_id(cluster, cate_id):
    for idx, each in enumerate(cluster):
        if cate_id in each:
            return each[0]
    return -1


def transform_cateid_into_cluster_id(cate_list, cluster):
    for idx in range(len(cate_list)):
        cluster_id = get_cluster_id(cluster, cate_list[idx].item())

        if cluster_id != -1:
            cate_list[idx] = cluster_id
    return cate_list


def trans_cluster_label(pred_pred_cate_list, gt_pred_cate_list, cluster):
    """
    transform the categories labels to cluster label for label overlapping avoiding
    :param pred_pair_mat: (subj_id, obj-id, cate-lable)
    :param gt_pair_mat:
    :return:
    """
    cluster_ref_pred_cate = transform_cateid_into_cluster_id(pred_pred_cate_list, cluster)
    cluster_ref_gt_cate = transform_cateid_into_cluster_id(gt_pred_cate_list, cluster)

    return cluster_ref_pred_cate, cluster_ref_gt_cate