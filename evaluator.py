import argparse
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
import random

class SOTAB_Evaluator:
    
    def __init__(self, ground_truth_filepath, submission_filepath, HAS_THRESHOLD, THRESHOLD):
        """
        ground_truth_filepath: filepath where csv with ground truth is located.
        submission_filepath: filepath where csv with submission is located.
        """
        self.ground_truth_filepath = ground_truth_filepath
        self.submission_filepath = submission_filepath
        self.HAS_THRESHOLD = HAS_THRESHOLD
        self.THRESHOLD = THRESHOLD
        self.submission_filepath = submission_filepath
    
    def _evaluate_per_class(self, gt_labels, predictions, cta_labels):
        precision, recall, macro_f1, _ = precision_recall_fscore_support(gt_labels, predictions, average='macro')
        micro_f1 = f1_score(gt_labels, predictions, average='micro')
        # Compute per-class scores
        per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
            gt_labels, predictions, average=None, labels=cta_labels
        )

        # Display per-class F1-scores
        print("\nF1-scores per class:")
        for label, prec, rec, f1 in zip(cta_labels, per_class_precision, per_class_recall, per_class_f1):
            print(f"Class: {label:20} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1-score: {f1:.4f}")

        # for label, f1 in zip(cta_labels, per_class_f1):
        #     print(f"Class: {label:20} | F1-score: {f1:.4f}")

        res = {
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'precision': precision,
            'recall': recall,
            'per_class_f1': dict(zip(cta_labels, per_class_f1))  # Optional: return as dict
        }
        print(res)

    def _evaluate(self):
        """
        Compare submitted annotations with ground truth annotations,
        and calculate precision, recall and macro-f1 and micro-f1 metrics.
        """

        gt = pd.read_csv(self.ground_truth_filepath)

        #sort approprietely 
        # gt['table'] = pd.Categorical(gt['table'], categories=gt['table'].unique(), ordered=True)
        # gt_sorted = gt.sort_values(by=['table', 'col'])
        # gt = gt_sorted


        submission = pd.read_csv(self.submission_filepath)


        if self.HAS_THRESHOLD:
            bad_mask = submission.apply(
                lambda col: col.astype(str).str.contains(r"\bnope\b", case=False, na=False)
            ).any(axis=1)


            # optional: inspect what will be removed
            print("Rows to delete from submission:")
            print(submission[bad_mask])

            # drop same rows from both
            submission = submission.loc[~bad_mask].reset_index(drop=True)
            gt = gt.loc[~bad_mask].reset_index(drop=True)


        gt_labels = gt['header'].tolist()
        #print(gt_labels)

        
        cta_labels = list(gt['header'].unique())

        # Number of predictions should equal the number of columns in the ground truth
        if len(submission) != len(gt):
            raise Exception("Some predictions are missing.")

        predictions = []

        sum=0
        for i in range(len(submission)):
            if submission.loc[i, 'header'] not in cta_labels:
                #print(submission.loc[i, 'header'])
                #submission.loc[i, 'header'] = random.choice(cta_labels)
                sum += 1
        print("+++",sum)

        for index, row in submission.iterrows():
            # # Prediction should be a label from the "sotab-cta-labels.txt" set.
            # if row['label'] not in cta_labels:
            #     raise Exception("Label out of label space used.")
            # else:
            predictions.append(row['header'])


        precision, recall, f1, _ = precision_recall_fscore_support(gt_labels, predictions, average='macro')
        micro_f1 = f1_score(gt_labels, predictions, average='micro')
        results = {
            'macro_f1': f1,
            'micro_f1': micro_f1,
            'precision': precision,
            'recall': recall
        }

        """
        Do something with your submitted file to come up
        with a score and a secondary score.

        if you want to report back an error to the user,
        then you can simply do :
          `raise Exception("YOUR-CUSTOM-ERROR")`

         You are encouraged to add as many validations as possible
         to provide meaningful feedback to your users
        """

        #self._evaluate_per_class(gt_labels, predictions, cta_labels)

        return results
   

class SOTAB_Evaluator_TopK:
    """
    Evaluator for CTA tables using Top-K predictions.
    'header' column in submission should contain top labels separated by '+',
    e.g., "Person+Telephone+Organization+Address+Email".
    """

    def __init__(self, ground_truth_filepath, submission_filepath, k=5):
        self.ground_truth_filepath = ground_truth_filepath
        self.submission_filepath = submission_filepath
        self.k = k  # Top-K parameter

    def _evaluate(self):
        gt = pd.read_csv(self.ground_truth_filepath)
        submission = pd.read_csv(self.submission_filepath)

        if len(submission) != len(gt):
            raise Exception("Some predictions are missing.")

        gt_labels = gt['header'].tolist()
        N = len(gt_labels)

        topk_predictions = []  # Stores a label for F1/precision/recall
        total_hits = 0

        for i in range(N):
            pred_string = submission.loc[i, 'header']
            labels = pred_string.split("+") if isinstance(pred_string, str) else []

            # Take only top-K predictions
            topk_labels = labels[:self.k]

            if gt_labels[i] in topk_labels:
                # Correct if GT is anywhere in top-K
                topk_predictions.append(gt_labels[i])
                total_hits += 1
            else:
                # Otherwise pick first predicted label (or empty)
                topk_predictions.append(labels[0] if labels else "")

        # Compute metrics exactly like original evaluator
        precision, recall, macro_f1, _ = precision_recall_fscore_support(
            gt_labels, topk_predictions, average='macro', zero_division=0
        )
        micro_f1 = f1_score(gt_labels, topk_predictions, average='micro', zero_division=0)

        results = {
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'precision': precision,
            'recall': recall,
            f'top{self.k}_accuracy': total_hits / N
        }

        return results