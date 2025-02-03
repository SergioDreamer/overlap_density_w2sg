import numpy as np
import torch
from ruptures import Binseg
from ruptures.exceptions import BadSegmentationParameters
from src.probe import PROBES

def setup_arms(K, detected_easy_partition, detected_hard_partition, detected_overlap_only_partition, x_strong_train, x_w2s_train_for_pseudolabeling, y_strong_train, overlap_density_list, number_of_samples_per_round, T, seed):
    arms = []
    for i in range(K):
        easy_indices = detected_easy_partition[i]
        hard_indices = detected_hard_partition[i]
        overlap_indices = detected_overlap_only_partition[i]
        # print(f"easy_indices: {easy_indices}")
        # print(f"hard_indices: {hard_indices}")
        # print(f"overlap_indices: {overlap_indices}")
        

        sample_indices = np.concatenate([easy_indices, hard_indices, overlap_indices])
        x_w2s_train_i = x_strong_train[sample_indices]
        x_w2s_train_for_pseudolabeling_i = x_w2s_train_for_pseudolabeling[sample_indices]
        y_w2s_train_i = y_strong_train[sample_indices]
        arms.append(Arm(x_w2s_train_i, x_w2s_train_for_pseudolabeling_i, y_w2s_train_i, overlap_density_list[i], number_of_samples_per_round, T, seed))
    return arms


def partition_indices_with_ratio(indices, ratio_list, seed):
    np.random.seed(seed)
    np.random.shuffle(indices)
    partition = []
    for ratio in ratio_list:
        n_sample = int(np.round(indices.shape[0] * ratio))
        partition.append(indices[:n_sample])
        indices = indices[n_sample:]
    return partition

def partition_indices_with_count(indices, count_list, seed):
    np.random.seed(seed)
    np.random.shuffle(indices)
    partition = []
    for n_sample in count_list:
        partition.append(indices[:n_sample])
        indices = indices[n_sample:]
    return partition



class Arm(object):
    def __init__(self, X, X_weak, y, overlap_density, num_samples_per_round, T, seed, easy_ratio=0.5, hard_ratio=0.5):
        self.seed = seed
        np.random.seed(seed)
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        self.X = X[indices]
        self.X_weak = X_weak[indices]
        self.y = y[indices]
        self.overlap_density = overlap_density
        self.easy_ratio = easy_ratio
        self.hard_ratio = hard_ratio
        self.num_samples_per_round = num_samples_per_round
        self.n_overlap_est = 0
        self.n_easy_est = 0
        self.n_hard_est = 0
        self.o_est = None
        self.count = 0
        self.T = T

    def update(self, partition_dict):
        n_easy_est = partition_dict["X_easy"].shape[0]
        n_hard_est = partition_dict["X_hard"].shape[0]
        n_overlap_est = partition_dict["X_overlap"].shape[0]
        self.n_easy_est += n_easy_est
        self.n_hard_est += n_hard_est
        self.n_overlap_est += n_overlap_est
        self.count += 1
        self.o_est = self.n_overlap_est / (self.n_easy_est + self.n_hard_est + self.n_overlap_est)
        
    def get_samples(self):
        X_sampled, X_weak_sampled, y_sampled = self.X[:self.num_samples_per_round], self.X_weak[:self.num_samples_per_round], self.y[:self.num_samples_per_round]
        self.X = self.X[self.num_samples_per_round:]
        self.X_weak = self.X_weak[self.num_samples_per_round:]
        self.y = self.y[self.num_samples_per_round:]
        return X_sampled, X_weak_sampled, y_sampled

    def compute_ucb(self):
        return self.o_est + np.sqrt(2 * np.log(self.T)) / self.count

class StrategyTracker(object):
    def __init__(self, X_test, y_test, probe_name, probe_cfg):
        self.count = 0
        self.X = []
        self.y = []
        self.y_pseudo = []
        
        self.X_detected_hard = [] # keep to help overlap detection
        self.X_detected_nonhard = [] # keep to help overlap detection
        self.X_detected_overlap_only = []
        self.y_detected_overlap_only = []
        self.y_pseudo_detected_overlap_only = []
        self.X_test = X_test
        self.y_test = y_test
        self.probe_name = probe_name
        self.probe_cfg = probe_cfg
    
    def update_samples(self, partition_dict):
        
        X = partition_dict["X"]
        y = partition_dict["y"]
        y_pseudo = partition_dict["y_pseudo"]
        X_detected_overlap_only = partition_dict["X_overlap"]
        y_detected_overlap_only = partition_dict["y_overlap"]
        y_pseudo_detected_overlap_only = partition_dict["y_pseudo_overlap"]
        X_detected_hard = partition_dict["X_hard"]
        X_detected_nonhard = partition_dict["X_nonhard"]

        self.X.append(X)
        self.y.append(y)
        self.y_pseudo.append(y_pseudo)
        self.X_detected_overlap_only.append(X_detected_overlap_only)
        self.y_detected_overlap_only.append(y_detected_overlap_only)
        self.y_pseudo_detected_overlap_only.append(y_pseudo_detected_overlap_only)
        self.X_detected_hard.append(X_detected_hard)
        self.X_detected_nonhard.append(X_detected_nonhard)

    def train_and_eval(self):
        
        if isinstance(self.X[0], torch.Tensor):
            x_w2s = torch.cat(self.X, device="cuda")
        else:
            x_w2s = np.concatenate(self.X, axis=0)
            x_w2s = torch.tensor(x_w2s, device="cuda")
        if isinstance(self.y_pseudo[0], torch.Tensor):
            y_w2s = torch.cat(self.y_pseudo, dim=0).to("cuda")
        else:
            y_w2s = np.concatenate(self.y_pseudo, axis=0)
            y_w2s = torch.tensor(y_w2s, device="cuda")
        
        if not isinstance(self.X_test, torch.Tensor):
            x_strong_test = torch.tensor(self.X_test, device="cuda")
        else:
            x_strong_test = self.X_test.to("cuda")
        if not isinstance(self.y_test, torch.Tensor):
            y_test = torch.tensor(self.y_test, device="cuda")
        else:
            y_test = self.y_test.to("cuda")
        

        w2s_overlap_probe = PROBES[self.probe_name](self.probe_cfg)
        # print('x_w2s.shape, y_w2s.shape, x_strong_test.shape, y_test.shape', x_w2s.shape, y_w2s.shape, x_strong_test.shape, y_test.shape)

        w2s_overlap_probe.fit(x_w2s, y_w2s)
        w2s_overlap_preds = w2s_overlap_probe.predict(x_strong_test)
        w2s_overlap_acc = (w2s_overlap_preds.round() == y_test).float().mean().item()
        return w2s_overlap_acc
    
    def train_and_eval_overlap_only(self):
        
        if isinstance(self.X_detected_overlap_only[0], torch.Tensor):
            x_w2s = torch.cat(self.X_detected_overlap_only, dim=0).to("cuda")
        else:
            x_w2s = np.concatenate(self.X_detected_overlap_only, axis=0)
            x_w2s = torch.tensor(x_w2s, device="cuda")
        if isinstance(self.y_pseudo_detected_overlap_only[0], torch.Tensor):
            y_w2s = torch.cat(self.y_pseudo_detected_overlap_only, dim=0).to("cuda")
        else:
            y_w2s = np.concatenate(self.y_pseudo_detected_overlap_only, axis=0)
            y_w2s = torch.tensor(y_w2s, device="cuda")
        
        if not isinstance(self.X_test, torch.Tensor):
            x_strong_test = torch.tensor(self.X_test, device="cuda")
        else:
            x_strong_test = self.X_test.to("cuda")
        if not isinstance(self.y_test, torch.Tensor):
            y_test = torch.tensor(self.y_test, device="cuda")
        else:
            y_test = self.y_test.to("cuda")
        

        w2s_overlap_probe = PROBES[self.probe_name](self.probe_cfg)
        # print('x_w2s.shape, y_w2s.shape, x_strong_test.shape, y_test.shape', x_w2s.shape, y_w2s.shape, x_strong_test.shape, y_test.shape)

        w2s_overlap_probe.fit(x_w2s, y_w2s)
        w2s_overlap_preds = w2s_overlap_probe.predict(x_strong_test)
        w2s_overlap_acc = (w2s_overlap_preds.round() == y_test).float().mean().item()
        return w2s_overlap_acc


    def get_est_overlap_ratio(self):
        return np.concatenate(self.X_detected_overlap_only).shape[0] / np.concatenate(self.X).shape[0]

    def get_gt_overlap_ratio(self):
        return np.concatenate(self.X_gt_overlap_only).shape[0] / np.concatenate(self.X).shape[0]

def detect_overlap_easy(X_nonhard, X_hard, X_nonhard_augmented, nonhard_indices):
    if isinstance(X_nonhard, torch.Tensor):
        X_nonhard = X_nonhard.detach().cpu().numpy()
    if isinstance(X_hard, torch.Tensor):
        X_hard = X_hard.detach().cpu().numpy()
    if isinstance(X_nonhard_augmented, torch.Tensor):
        X_nonhard_augmented = X_nonhard_augmented.detach().cpu().numpy()
    overlap_scores = (X_nonhard @ X_hard.T).max(axis=1)
    overlap_scores_augmented = (X_nonhard_augmented @ X_hard.T).max(axis=1)
    
    # Apply change point detection to decide threshold for align scores

    # Perform change point detection
    try:
        sorted_overlap_scores_augmented = np.sort(overlap_scores_augmented)
        model = Binseg(model="l2").fit(sorted_overlap_scores_augmented.reshape(-1, 1))
        change_points = model.predict(n_bkps=1)[0]
        overlap_score_threshold = sorted_overlap_scores_augmented[change_points]
    except Exception as e:
        print(e)
        print("Change point detection failed. Using quantile 0.5 as threshold in overlap scores.")
        overlap_score_threshold = np.quantile(overlap_scores_augmented, 0.5)

    # Use the detected change point as the threshold
    detected_overlap_indices = np.where(overlap_scores >= overlap_score_threshold)[0]
    detected_easy_indices = np.where(overlap_scores < overlap_score_threshold)[0]

    easy_indices = nonhard_indices[detected_easy_indices]
    overlap_indices = nonhard_indices[detected_overlap_indices]

    detected_easy_indices = easy_indices
    
    detected_overlap_indices = overlap_indices
    return detected_easy_indices, detected_overlap_indices

def detect_hard_nonhard(X, y_pseudo_proba):
    # print('y_pseudo_proba', y_pseudo_proba)
    confidence_scores = np.abs(y_pseudo_proba - 0.5) * 2
    # print('confidence_scores', confidence_scores)

    sorted_confidence = np.sort(confidence_scores)

    # Fix | Check if sorted_confidence is empty
    if sorted_confidence.size == 0:
        raise ValueError("sorted_confidence is empty. Cannot detect change points.")
    
    # Perform change point detection
    try:
        model = Binseg(model="l2").fit(sorted_confidence.reshape(-1, 1))
        change_points = model.predict(n_bkps=1)

        # Fix
        if len(change_points) == 0 or change_points[0] >= len(sorted_confidence):
            raise ValueError("Invalid change points detected.")
        
        # Use the detected change point as the threshold
        confidence_score_threshold = sorted_confidence[change_points[0]]
    except BadSegmentationParameters:
        raise ValueError("Bad segmentation parameters. Cannot detect change points.")
    except Exception as e:
        print(e)
        print("Change point detection failed. Using median as threshold in confidence scores.")
        confidence_score_threshold = np.quantile(confidence_scores, 0.5)


    detected_hard_indices = np.where(confidence_scores < confidence_score_threshold)[0]
    detected_nonhard_indices = np.where(confidence_scores >= confidence_score_threshold)[0]
    return detected_hard_indices, detected_nonhard_indices

def detect_easy_hard_overlap(X, y_pseudo_proba, tracker):
    if isinstance(y_pseudo_proba, torch.Tensor):
        y_pseudo_proba = y_pseudo_proba.detach().cpu().numpy()

    detected_hard_indices, detected_nonhard_indices = detect_hard_nonhard(X, y_pseudo_proba)
    X_hard, X_nonhard = X[detected_hard_indices], X[detected_nonhard_indices]
    X_hard = torch.tensor(X_hard, device="cuda")
    X_nonhard = torch.tensor(X_nonhard, device="cuda")

    if len(tracker.X_detected_hard) > 0:
        X_detected_hard_history = tracker.X_detected_hard
        if isinstance(X_detected_hard_history[0], torch.Tensor):
            X_detected_hard_history = torch.concat(X_detected_hard_history, axis=0)
        else:
            X_detected_hard_history = np.concatenate(tracker.X_detected_hard, axis=0)
            X_detected_hard_history = torch.tensor(X_detected_hard_history, device="cuda")
        X_hard_augmented = torch.concat([X_hard, X_detected_hard_history], axis=0)
    else:
        X_hard_augmented = X_hard

    if len(tracker.X_detected_nonhard) > 0:
        X_detected_nonhard_history = tracker.X_detected_nonhard
        if isinstance(X_detected_nonhard_history[0], torch.Tensor):
            X_detected_nonhard_history = torch.concat(X_detected_nonhard_history, axis=0)
        else:
            X_detected_nonhard_history = np.concatenate(tracker.X_detected_nonhard, axis=0)
            X_detected_nonhard_history = torch.tensor(X_detected_nonhard_history, device="cuda")
        X_nonhard_augmented = torch.concat([X_nonhard, X_detected_nonhard_history], axis=0)
    else:
        X_nonhard_augmented = X_nonhard
    detected_easy_indices, detected_overlap_indices = detect_overlap_easy(X_nonhard, X_hard_augmented, X_nonhard_augmented, detected_nonhard_indices)

    return_dict = {
        'detected_easy_indices': detected_easy_indices,
        'detected_hard_indices': detected_hard_indices,
        'detected_nonhard_indices': detected_nonhard_indices,
        'detected_overlap_indices': detected_overlap_indices,
    }
    return return_dict

def detect_and_partition(X, y, y_pseudo, tracker):
    detected_indices = detect_easy_hard_overlap(X, y_pseudo, tracker)
    detected_easy_indices = detected_indices['detected_easy_indices']
    detected_hard_indices = detected_indices['detected_hard_indices']
    detected_nonhard_indices = detected_indices['detected_nonhard_indices']
    detected_overlap_indices = detected_indices['detected_overlap_indices']

    X_easy = X[detected_easy_indices]
    X_hard = X[detected_hard_indices]
    X_nonhard = X[detected_nonhard_indices]
    X_overlap = X[detected_overlap_indices]
    

    y_easy = y[detected_easy_indices]
    y_hard = y[detected_hard_indices]
    y_overlap = y[detected_overlap_indices]
    y_nonhard = y[detected_nonhard_indices]
    y_pseudo_easy = y_pseudo[detected_easy_indices]
    y_pseudo_hard = y_pseudo[detected_hard_indices]
    y_pseudo_overlap = y_pseudo[detected_overlap_indices]
    y_pseudo_nonhard = y_pseudo[detected_nonhard_indices]

    return_dict = {
        "X": X,
        "y": y,
        "y_pseudo": y_pseudo,
        "X_easy": X_easy,
        "X_hard": X_hard,
        "X_overlap": X_overlap,
        "X_nonhard": X_nonhard,
        "y_pseudo_easy": y_pseudo_easy,
        "y_pseudo_hard": y_pseudo_hard,
        "y_pseudo_overlap": y_pseudo_overlap,
        "y_pseudo_nonhard": y_pseudo_nonhard,
        "y_hard": y_hard,
        "y_easy": y_easy,
        "y_overlap": y_overlap,
        "detected_easy_indices": detected_easy_indices,
        "detected_hard_indices": detected_hard_indices,
        "detected_overlap_indices": detected_overlap_indices,
        "detected_nonhard_indices": detected_nonhard_indices,
    }
    return return_dict
