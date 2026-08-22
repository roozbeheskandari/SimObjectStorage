
import os
import math
import time
import hashlib
import random
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional
import multiprocessing
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


RUN_MODE = os.environ.get("SIM_RUN_MODE", "paper")  # "quick" or "paper"
RANDOM_SEED = int(os.environ.get("SIM_SEED", "42")) #thhis is default SEED in ML
OUTPUT_DIR = os.environ.get("SIM_OUTDIR", "outputs_final") # based on ur OS create Directory for Results
os.makedirs(OUTPUT_DIR, exist_ok=True)

if RUN_MODE == "quick":
    OBJECT_COUNTS = [2_000, 5_000]
    DUP_RATIOS = [0.1, 0.3, 0.5]
    REPEATS = 2
else:
    OBJECT_COUNTS = [1_000_000, 20_000_000, 50_000_000] #change these values for test
    DUP_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5]
    REPEATS = 3 //Change Repeat

QUERY_MULTIPLIER = 1.0
TRAIN_RATIO = 0.70
LEARNED_THRESHOLD = 0.50

# Bloom parameters
BLOOM_BITS_PER_ITEM = 10
BLOOM_NUM_HASHES = 7

# Cuckoo Filter parameters
CUCKOO_BUCKET_SIZE = 4
CUCKOO_FINGERPRINT_BITS = 12
CUCKOO_MAX_KICKS = 500
CUCKOO_LOAD_FACTOR = 0.95

# Synthetic latency model constants (ms)
LAT_HASH_MS = 0.003
LAT_BLOOM_MS = 0.006
LAT_CUCKOO_MS = 0.008
LAT_CLASSIFIER_MS = 0.015
LAT_METADATA_MS = 0.050
LAT_STORAGE_WRITE_MS = 0.120
LAT_NETWORK_MS = 0.030
LAT_FEATURE_BUILD_MS = 0.004

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)



def run_simulation_task(task_args):

    repeat_id, object_count, duplication_ratio, methods_list = task_args
    results = []

    sim_instance = SimulationV4()
    workload = Workload(object_count, duplication_ratio, seed=RANDOM_SEED + repeat_id)
    items = workload.generate()
    
    for method_name, method_key in methods_list:
        row = sim_instance._run_one(method_name, method_key, items, repeat_id, object_count, duplication_ratio)
        results.append(row)
        
    return results

def sha256_int(x: int) -> int:
    return int(hashlib.sha256(str(x).encode("utf-8")).hexdigest(), 16)


def stable_hash(x: int, seed: int) -> int:
    payload = f"{x}-{seed}".encode("utf-8")
    return int(hashlib.blake2b(payload, digest_size=8).hexdigest(), 16)


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def mean_or_zero(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def percentile_or_zero(values: List[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def format_million(n: int) -> str:
    return f"{n/1_000_000:.2f}M"


def workload_signature(object_count: int, duplication_ratio: float) -> str:
    return f"N={object_count}|D={duplication_ratio:.3f}"


@dataclass
class WorkloadItem:
    key: int
    is_duplicate: int
    features: Tuple[float, float, float, float]


class Workload:
    def __init__(self, object_count: int, duplication_ratio: float, seed: int = RANDOM_SEED):
        self.object_count = int(object_count)
        self.duplication_ratio = float(duplication_ratio)
        self.seed = int(seed)
        self.items: List[WorkloadItem] = []

    def generate(self) -> List[WorkloadItem]:
        rng = np.random.default_rng(self.seed + self.object_count + int(self.duplication_ratio * 10_000))
        n_dup = int(round(self.object_count * self.duplication_ratio))
        n_unique = self.object_count - n_dup

        unique_keys = rng.choice(np.arange(1, self.object_count * 20 + 1), size=max(n_unique, 1), replace=False)
        duplicates = []
        if n_dup > 0 and len(unique_keys) > 0:
            dup_sources = rng.choice(unique_keys, size=n_dup, replace=True)
            duplicates = dup_sources.tolist()

        all_keys = unique_keys.tolist() + duplicates
        rng.shuffle(all_keys)

        self.items = []
        seen = set()
        for key in all_keys:
            is_dup = 1 if key in seen else 0
            seen.add(key)
            feats = self._build_features(key, is_dup, rng)
            self.items.append(WorkloadItem(int(key), int(is_dup), feats))
        return self.items

    @staticmethod
    def _build_features(key: int, is_dup: int, rng: np.random.Generator) -> Tuple[float, float, float, float]:
        h = sha256_int(key)
        x1 = (h & 0xFFFF) / 65535.0
        x2 = ((h >> 16) & 0xFFFF) / 65535.0
        x3 = ((h >> 32) & 0xFFFF) / 65535.0
        x4 = ((h >> 48) & 0xFFFF) / 65535.0
        noise = rng.normal(0.0, 0.03, size=4)
        base = np.array([x1, x2, x3, x4], dtype=float)
        if is_dup:
            base = base * 0.72 + 0.18
        else:
            base = base * 0.95 + 0.02
        base = np.clip(base + noise, 0.0, 1.0)
        return (
            float(base[0]),
            float(base[1]),
            float(base[2]),
            float(base[3]),
)


class BloomFilter:
    def __init__(self, capacity: int, bits_per_item: int = BLOOM_BITS_PER_ITEM, num_hashes: int = BLOOM_NUM_HASHES):
        self.capacity = max(1, int(capacity))
        self.bits_per_item = int(bits_per_item)
        self.num_hashes = int(num_hashes)
        self.size = max(8, self.capacity * self.bits_per_item)
        self.bits = np.zeros(self.size, dtype=np.uint8)
        self.item_count = 0

    def _hashes(self, key: int):
        h1 = stable_hash(key, 11)
        h2 = stable_hash(key, 29) | 1
        for i in range(self.num_hashes):
            yield (h1 + i * h2) % self.size

    def add(self, key: int):
        for idx in self._hashes(key):
            self.bits[idx] = 1
        self.item_count += 1

    def __contains__(self, key: int) -> bool:
        return all(self.bits[idx] for idx in self._hashes(key))

    @property
    def memory_bits(self) -> int:
        return int(self.bits.size)


class CuckooFilter:
    def __init__(self, capacity: int, bucket_size: int = CUCKOO_BUCKET_SIZE, fingerprint_bits: int = CUCKOO_FINGERPRINT_BITS, max_kicks: int = CUCKOO_MAX_KICKS):
        self.capacity = max(1, int(capacity))
        self.bucket_size = int(bucket_size)
        self.fingerprint_bits = int(fingerprint_bits)
        self.max_kicks = int(max_kicks)
        buckets = int(math.ceil(self.capacity / (self.bucket_size * CUCKOO_LOAD_FACTOR)))
        self.num_buckets = max(8, 1 << int(math.ceil(math.log2(buckets))))
        self.table: List[List[int]] = [[] for _ in range(self.num_buckets)]
        self.item_count = 0

    def _fingerprint(self, key: int) -> int:
        fp_mask = (1 << self.fingerprint_bits) - 1
        fp = stable_hash(key, 101) & fp_mask
        return fp or 1

    def _index1(self, key: int) -> int:
        return stable_hash(key, 202) % self.num_buckets

    def _index2(self, i1: int, fp: int) -> int:
        return (i1 ^ (stable_hash(fp, 303) % self.num_buckets)) % self.num_buckets

    def add(self, key: int) -> bool:
        fp = self._fingerprint(key)
        i1 = self._index1(key)
        i2 = self._index2(i1, fp)
        if len(self.table[i1]) < self.bucket_size:
            self.table[i1].append(fp)
            self.item_count += 1
            return True
        if len(self.table[i2]) < self.bucket_size:
            self.table[i2].append(fp)
            self.item_count += 1
            return True

        i = i1 if random.random() < 0.5 else i2
        cur_fp = fp
        for _ in range(self.max_kicks):
            slot = random.randrange(len(self.table[i]))
            self.table[i][slot], cur_fp = cur_fp, self.table[i][slot]
            i = self._index2(i, cur_fp)
            if len(self.table[i]) < self.bucket_size:
                self.table[i].append(cur_fp)
                self.item_count += 1
                return True
        return False

    def __contains__(self, key: int) -> bool:
        fp = self._fingerprint(key)
        i1 = self._index1(key)
        i2 = self._index2(i1, fp)
        return fp in self.table[i1] or fp in self.table[i2]

    @property
    def memory_bits(self) -> int:
        return int(self.num_buckets * self.bucket_size * self.fingerprint_bits)


class LearnedBinaryFilter:
    def __init__(self, base_filter_type: str = "bloom"):
        self.base_filter_type = base_filter_type
        self.model = LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced")
        self.backup_filter = None
        self.train_metrics: Dict[str, float] = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "train_time_s": 0.0,
        }
        self.feature_dim = 4

    @staticmethod
    def build_features(keys: List[int], duplication_ratio: float) -> np.ndarray:
        feats = []
        for key in keys:
            h = sha256_int(int(key))
            x = [
                (h & 0xFFFF) / 65535.0,
                ((h >> 16) & 0xFFFF) / 65535.0,
                ((h >> 32) & 0xFFFF) / 65535.0,
                ((h >> 48) & 0xFFFF) / 65535.0,
            ]
            x.append(float(duplication_ratio))
            feats.append(x)
        arr = np.asarray(feats, dtype=float)
        return arr[:, :4]

    def fit(self, keys: List[int], labels: List[int], duplication_ratio: float, capacity: int):
        X = self.build_features(keys, duplication_ratio)
        y = np.asarray(labels, dtype=int)
        if len(np.unique(y)) < 2:
            raise ValueError("LearnedBinaryFilter requires both classes in training data.")

        t0 = time.perf_counter()
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=max(0.2, 1.0 - TRAIN_RATIO), random_state=RANDOM_SEED, stratify=y
        )
        self.model.fit(X_train, y_train)
        val_pred = self.model.predict(X_val)
        t1 = time.perf_counter()

        self.train_metrics = {
            "accuracy": float(accuracy_score(y_val, val_pred)),
            "precision": float(precision_score(y_val, val_pred, zero_division=0)),
            "recall": float(recall_score(y_val, val_pred, zero_division=0)),
            "f1": float(f1_score(y_val, val_pred, zero_division=0)),
            "train_time_s": float(t1 - t0),
        }

        if self.base_filter_type == "bloom":
            self.backup_filter = BloomFilter(capacity=capacity)
        else:
            self.backup_filter = CuckooFilter(capacity=capacity)

        for key, label in zip(keys, labels):
            if int(label) == 1:
                self.backup_filter.add(int(key))
        return self

    def predict_prob(self, keys: List[int], duplication_ratio: float) -> np.ndarray:
        X = self.build_features(keys, duplication_ratio)
        return self.model.predict_proba(X)[:, 1]

    def __contains__(self, key: int) -> bool:
        return key in self.backup_filter if self.backup_filter is not None else False

    @property
    def memory_bits(self) -> int:
        if self.backup_filter is None:
            return 0
        return int(self.backup_filter.memory_bits)


@dataclass
class ResultRow:
    method: str
    method_group: str
    repeat_id: int
    object_count: int
    duplication_ratio: float
    query_count: int

    tp: int
    tn: int
    fp: int
    fn: int

    accuracy: float
    precision: float
    recall: float
    fpr: float
    fnr: float
    specificity: float
    npv: float
    f1: float

    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_qps: float

    cpu_ops: float
    cpu_cost_norm: float
    metadata_ops: float
    metadata_ops_norm: float
    bandwidth_kb: float
    bandwidth_per_object_kb: float
    memory_bits: float
    memory_kb: float
    dedup_ratio: float
    dedup_rate: float

    insert_ops: float
    lookup_ops: float
    relocation_ops: float
    hash1_ops: float
    hash2_ops: float
    false_positive_ops: float
    true_positive_ops: float
    true_negative_ops: float
    false_negative_ops: float

    learned_train_accuracy: float = np.nan
    learned_train_precision: float = np.nan
    learned_train_recall: float = np.nan
    learned_train_f1: float = np.nan
    train_time_s: float = np.nan

    backup_memory_kb: float = 0.0
    model_memory_kb: float = 0.0

    extra_note: str = ""


def compute_confusion(y_true: List[int], y_pred: List[int]) -> Tuple[int, int, int, int]:
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tp = int(cm[0, 0])
    fn = int(cm[0, 1])
    fp = int(cm[1, 0])
    tn = int(cm[1, 1])
    return tp, tn, fp, fn


def compute_metrics(y_true: List[int], y_pred: List[int], latency_ms: List[float]) -> Dict[str, float]:
    tp, tn, fp, fn = compute_confusion(y_true, y_pred)
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)
    spec = safe_div(tn, tn + fp)
    npv = safe_div(tn, tn + fn)
    f1 = safe_div(2 * prec * rec, prec + rec)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "fpr": fpr,
        "fnr": fnr,
        "specificity": spec,
        "npv": npv,
        "f1": f1,
        "latency_mean_ms": mean_or_zero(latency_ms),
        "latency_p50_ms": percentile_or_zero(latency_ms, 50),
        "latency_p95_ms": percentile_or_zero(latency_ms, 95),
        "latency_p99_ms": percentile_or_zero(latency_ms, 99),
    }



class SimulationV4:
    def __init__(self):
        self.results: List[ResultRow] = []
        self.methods = [
            ("Baseline", "baseline"),
            ("Bloom", "bloom"),
            ("Cuckoo", "cuckoo"),
            ("Learned-Bloom", "learned_bloom"),
            ("Learned-Cuckoo", "learned_cuckoo"),
        ]

    #def run(self) -> pd.DataFrame:
          #for repeat_id in range(1, REPEATS + 1):
            #for object_count in OBJECT_COUNTS:
               # for duplication_ratio in DUP_RATIOS:
                   # workload = Workload(object_count, duplication_ratio, seed=RANDOM_SEED + repeat_id)
                    #items = workload.generate()
                   # for method_name, method_key in self.methods:
                      #  row = self._run_one(method_name, method_key, items, repeat_id, object_count, duplication_ratio)
                       # self.results.append(row)
        #return pd.DataFrame([asdict(r) for r in self.results])

    def run(self) -> pd.DataFrame:
        tasks = []
        for repeat_id in range(1, REPEATS + 1):
            for object_count in OBJECT_COUNTS:
                for duplication_ratio in DUP_RATIOS:
                   
                    tasks.append((repeat_id, object_count, duplication_ratio, self.methods))
        
        num_cores = multiprocessing.cpu_count()
        print(f"[*] Starting simulation on {num_cores} CPU cores...")
        
        with multiprocessing.Pool(processes=num_cores) as pool:
            results_nested = pool.map(run_simulation_task, tasks)  
        
        for result_batch in results_nested:
            self.results.extend(result_batch)
            
        return pd.DataFrame([asdict(r) for r in self.results])
    

    def _run_one(self, method_name: str, method_key: str, items: List[WorkloadItem], repeat_id: int, object_count: int, duplication_ratio: float) -> ResultRow:
        query_count = int(max(1, round(len(items) * QUERY_MULTIPLIER)))
        query_items = items[:query_count]
        y_true = [it.is_duplicate for it in query_items]
        latencies = []

        cpu_ops = 0.0
        metadata_ops = 0.0
        bandwidth_kb = 0.0
        relocation_ops = 0.0
        hash1_ops = 0.0
        hash2_ops = 0.0
        false_positive_ops = 0.0
        true_positive_ops = 0.0
        true_negative_ops = 0.0
        false_negative_ops = 0.0
        insert_ops = 0.0
        lookup_ops = 0.0
        backup_memory_kb = 0.0
        model_memory_kb = 0.0
        train_metrics = {"accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan, "train_time_s": np.nan}

        if method_key == "baseline":
            seen = set()
            y_pred = []
            for it in query_items:
                t0 = time.perf_counter()
                is_dup = 1 if it.key in seen else 0
                seen.add(it.key)
                t1 = time.perf_counter()
                y_pred.append(is_dup)
                latencies.append((t1 - t0) * 1000.0 + LAT_HASH_MS + LAT_METADATA_MS)
                cpu_ops += 1.0
                metadata_ops += 1.0
                bandwidth_kb += 0.03
                lookup_ops += 1.0
                true_positive_ops += float(is_dup == 1 and it.is_duplicate == 1)
                true_negative_ops += float(is_dup == 0 and it.is_duplicate == 0)
                false_positive_ops += float(is_dup == 1 and it.is_duplicate == 0)
                false_negative_ops += float(is_dup == 0 and it.is_duplicate == 1)

        elif method_key == "bloom":
            bf = BloomFilter(capacity=object_count)
            seen = set()
            for it in items:
                if it.is_duplicate == 0:
                    bf.add(it.key)
                    seen.add(it.key)
                    insert_ops += 1.0
            y_pred = []
            for it in query_items:
                t0 = time.perf_counter()
                pred = 1 if it.key in bf else 0
                t1 = time.perf_counter()
                y_pred.append(pred)
                latencies.append((t1 - t0) * 1000.0 + LAT_BLOOM_MS + LAT_HASH_MS * BLOOM_NUM_HASHES)
                cpu_ops += float(BLOOM_NUM_HASHES)
                metadata_ops += 1.0
                bandwidth_kb += 0.01
                hash1_ops += 1.0
                hash2_ops += float(BLOOM_NUM_HASHES - 1)
                true_positive_ops += float(pred == 1 and it.is_duplicate == 1)
                true_negative_ops += float(pred == 0 and it.is_duplicate == 0)
                false_positive_ops += float(pred == 1 and it.is_duplicate == 0)
                false_negative_ops += float(pred == 0 and it.is_duplicate == 1)
            backup_memory_kb = bf.memory_bits / 8.0 / 1024.0


        elif method_key == "cuckoo":
            cf = CuckooFilter(capacity=object_count)
            for it in items:
                if it.is_duplicate == 0:
                    cf.add(it.key)
                    insert_ops += 1.0
            y_pred = []
            for it in query_items:
                t0 = time.perf_counter()
                pred = 1 if it.key in cf else 0
                t1 = time.perf_counter()
                y_pred.append(pred)
                latencies.append((t1 - t0) * 1000.0 + LAT_CUCKOO_MS + LAT_HASH_MS * 2)
                cpu_ops += 2.0
                metadata_ops += 1.0
                bandwidth_kb += 0.015
                relocation_ops += 0.002
                hash1_ops += 1.0
                hash2_ops += 1.0
                true_positive_ops += float(pred == 1 and it.is_duplicate == 1)
                true_negative_ops += float(pred == 0 and it.is_duplicate == 0)
                false_positive_ops += float(pred == 1 and it.is_duplicate == 0)
                false_negative_ops += float(pred == 0 and it.is_duplicate == 1)
            backup_memory_kb = cf.memory_bits / 8.0 / 1024.0


        elif method_key == "learned_bloom":
            X_keys = [it.key for it in items]
            y_labels = [it.is_duplicate for it in items]
            learner = LearnedBinaryFilter(base_filter_type="bloom")
            learner.fit(X_keys, y_labels, duplication_ratio, capacity=object_count)
            train_metrics = learner.train_metrics.copy()
            y_pred = []
            probs = learner.predict_prob([it.key for it in query_items], duplication_ratio)
            for it, p in zip(query_items, probs):
                t0 = time.perf_counter()
                pred = 1 if p >= LEARNED_THRESHOLD else 0
                if pred == 0 and learner.backup_filter is not None:
                    if it.key in learner.backup_filter:
                        pred = 1 #if it.key in learner.backup_filter else pred
                t1 = time.perf_counter()
                y_pred.append(pred)
                latencies.append((t1 - t0) * 1000.0 + LAT_CLASSIFIER_MS + LAT_FEATURE_BUILD_MS)
                cpu_ops += 4.0
                metadata_ops += 1.0
                bandwidth_kb += 0.02
                hash1_ops += 1.0
                hash2_ops += 1.0
                true_positive_ops += float(pred == 1 and it.is_duplicate == 1)
                true_negative_ops += float(pred == 0 and it.is_duplicate == 0)
                false_positive_ops += float(pred == 1 and it.is_duplicate == 0)
                false_negative_ops += float(pred == 0 and it.is_duplicate == 1)
            model_memory_kb = (len(learner.model.coef_.ravel()) + len(learner.model.intercept_)) * 8.0 / 1024.0
            backup_memory_kb = learner.memory_bits / 8.0 / 1024.0
            cpu_ops += 10.0
            metadata_ops += 5.0


        elif method_key == "learned_cuckoo":
            X_keys = [it.key for it in items]
            y_labels = [it.is_duplicate for it in items]
            learner = LearnedBinaryFilter(base_filter_type="cuckoo")
            learner.fit(X_keys, y_labels, duplication_ratio, capacity=object_count)
            train_metrics = learner.train_metrics.copy()
            y_pred = []
            probs = learner.predict_prob([it.key for it in query_items], duplication_ratio)
            for it, p in zip(query_items, probs):
                t0 = time.perf_counter()
                pred = 1 if p >= LEARNED_THRESHOLD else 0
                if pred == 1 and learner.backup_filter is not None:
                    if it.key in learner.backup_filter:
                        pred = 1 # if it.key in learner.backup_filter else pred
                t1 = time.perf_counter()
                y_pred.append(pred)
                latencies.append((t1 - t0) * 1000.0 + LAT_CLASSIFIER_MS + LAT_FEATURE_BUILD_MS)
                cpu_ops += 4.0
                metadata_ops += 1.0
                bandwidth_kb += 0.022
                relocation_ops += 0.003
                hash1_ops += 1.0
                hash2_ops += 1.0
                true_positive_ops += float(pred == 1 and it.is_duplicate == 1)
                true_negative_ops += float(pred == 0 and it.is_duplicate == 0)
                false_positive_ops += float(pred == 1 and it.is_duplicate == 0)
                false_negative_ops += float(pred == 0 and it.is_duplicate == 1)
            model_memory_kb = (len(learner.model.coef_.ravel()) + len(learner.model.intercept_)) * 8.0 / 1024.0
            backup_memory_kb = learner.memory_bits / 8.0 / 1024.0
            cpu_ops += 10.0
            metadata_ops += 5.0

        else:
            raise ValueError(f"Unknown method: {method_key}")

        metrics = compute_metrics(y_true, y_pred, latencies)

        dedup_ratio = safe_div(true_positive_ops + true_negative_ops, len(query_items))
        dedup_rate = safe_div(true_positive_ops, true_positive_ops + false_negative_ops)
        memory_kb = backup_memory_kb + model_memory_kb
        cpu_cost_norm = cpu_ops / max(1.0, query_count)
        metadata_ops_norm = metadata_ops / max(1.0, query_count)
        bandwidth_per_object_kb = bandwidth_kb / max(1.0, query_count)
        throughput_qps = safe_div(1000.0, metrics["latency_mean_ms"]) if metrics["latency_mean_ms"] > 0 else 0.0

        return ResultRow(
            method=method_name,
            method_group=method_key,
            repeat_id=repeat_id,
            object_count=object_count,
            duplication_ratio=duplication_ratio,
            query_count=query_count,
            tp=int(metrics["tp"]),
            tn=int(metrics["tn"]),
            fp=int(metrics["fp"]),
            fn=int(metrics["fn"]),
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            fpr=metrics["fpr"],
            fnr=metrics["fnr"],
            specificity=metrics["specificity"],
            npv=metrics["npv"],
            f1=metrics["f1"],
            latency_mean_ms=metrics["latency_mean_ms"],
            latency_p50_ms=metrics["latency_p50_ms"],
            latency_p95_ms=metrics["latency_p95_ms"],
            latency_p99_ms=metrics["latency_p99_ms"],
            throughput_qps=throughput_qps,
            cpu_ops=cpu_ops,
            cpu_cost_norm=cpu_cost_norm,
            metadata_ops=metadata_ops,
            metadata_ops_norm=metadata_ops_norm,
            bandwidth_kb=bandwidth_kb,
            bandwidth_per_object_kb=bandwidth_per_object_kb,
            memory_bits=memory_kb * 8.0 * 1024.0,
            memory_kb=memory_kb,
            dedup_ratio=dedup_ratio,
            dedup_rate=dedup_rate,
            insert_ops=insert_ops,
            lookup_ops=lookup_ops,
            relocation_ops=relocation_ops,
            hash1_ops=hash1_ops,
            hash2_ops=hash2_ops,
            false_positive_ops=false_positive_ops,
            true_positive_ops=true_positive_ops,
            true_negative_ops=true_negative_ops,
            false_negative_ops=false_negative_ops,
            learned_train_accuracy=float(train_metrics.get("accuracy", np.nan)),
            learned_train_precision=float(train_metrics.get("precision", np.nan)),
            learned_train_recall=float(train_metrics.get("recall", np.nan)),
            learned_train_f1=float(train_metrics.get("f1", np.nan)),
            train_time_s=float(train_metrics.get("train_time_s", np.nan)),
            backup_memory_kb=backup_memory_kb,
            model_memory_kb=model_memory_kb,
            extra_note=workload_signature(object_count, duplication_ratio),
        )



def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["method", "method_group", "object_count", "duplication_ratio"]
    agg = {
        "repeat_id": "count",
        "tp": "mean",
        "tn": "mean",
        "fp": "mean",
        "fn": "mean",
        "accuracy": "mean",
        "precision": "mean",
        "recall": "mean",
        "fpr": "mean",
        "fnr": "mean",
        "specificity": "mean",
        "npv": "mean",
        "f1": "mean",
        "latency_mean_ms": "mean",
        "latency_p50_ms": "mean",
        "latency_p95_ms": "mean",
        "latency_p99_ms": "mean",
        "throughput_qps": "mean",
        "cpu_ops": "mean",
        "cpu_cost_norm": "mean",
        "metadata_ops": "mean",
        "metadata_ops_norm": "mean",
        "bandwidth_kb": "mean",
        "bandwidth_per_object_kb": "mean",
        "memory_bits": "mean",
        "memory_kb": "mean",
        "dedup_ratio": "mean",
        "dedup_rate": "mean",
        "insert_ops": "mean",
        "lookup_ops": "mean",
        "relocation_ops": "mean",
        "hash1_ops": "mean",
        "hash2_ops": "mean",
        "false_positive_ops": "mean",
        "true_positive_ops": "mean",
        "true_negative_ops": "mean",
        "false_negative_ops": "mean",
        "learned_train_accuracy": "mean",
        "learned_train_precision": "mean",
        "learned_train_recall": "mean",
        "learned_train_f1": "mean",
        "train_time_s": "mean",
        "backup_memory_kb": "mean",
        "model_memory_kb": "mean",
    }
    out = df.groupby(group_cols, as_index=False).agg(agg)
    out = out.rename(columns={"repeat_id": "n_repeats"})
    return out


def apply_ieee_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
        "grid.linestyle": "--",
        "grid.alpha": 0.55,
    })


def grouped_bar_plot(df: pd.DataFrame, metric: str, ylabel: str, title: str, filename: str, x_key: str = "object_count"):
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x_values = sorted(df[x_key].unique())
    methods = ["Baseline", "Bloom", "Cuckoo", "Learned-Bloom", "Learned-Cuckoo"]

    bar_w = 0.16
    x = np.arange(len(x_values))
    offsets = np.linspace(-2 * bar_w, 2 * bar_w, len(methods))

    style_map = {
        "Baseline": {"color": "#f2f2f2", "hatch": "//"},
        "Bloom": {"color": "#d9d9d9", "hatch": "\\\\"},
        "Cuckoo": {"color": "#bdbdbd", "hatch": "||"},
        "Learned-Bloom": {"color": "#969696", "hatch": "--"},
        "Learned-Cuckoo": {"color": "#737373", "hatch": "xx"},
    }

    for idx, method in enumerate(methods):
        ys = []
        for xv in x_values:
            sub = df[(df[x_key] == xv) & (df["method"] == method)]
            ys.append(float(sub[metric].mean()) if len(sub) else 0.0)
        ax.bar(
            x + offsets[idx],
            ys,
            width=bar_w,
            label=method,
            color=style_map[method]["color"],
            hatch=style_map[method]["hatch"],
            edgecolor="black",
            linewidth=0.8,
        )

    ax.set_xlabel("Number of items (Millions)" if x_key == "object_count" else "Duplication ratio")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if x_key == "object_count":
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v/1_000_000:.2f}" for v in x_values])
    else:
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:.2f}" for v in x_values])

    ax.grid(axis="y")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.34), frameon=True, framealpha=1.0)
    #fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.70))
    fig.subplots_adjust(top=0.72, bottom=0.15, left=0.10, right=0.95)
    out_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def grouped_bar_plot_by_dup_ratio(df: pd.DataFrame, metric: str, ylabel: str, title: str, filename: str):
    return grouped_bar_plot(df, metric, ylabel, title, filename, x_key="duplication_ratio")


def plot_train_test_metrics(df: pd.DataFrame):
    learned = df[df["method_group"].isin(["learned_bloom", "learned_cuckoo"])].copy()
    if learned.empty:
        return []
    apply_ieee_style()
    metrics = ["learned_train_accuracy", "learned_train_precision", "learned_train_recall", "learned_train_f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    methods = ["Learned-Bloom", "Learned-Cuckoo"]
    files = []

    for m, lab in zip(metrics, labels):
        fig, ax = plt.subplots(figsize=(6.8, 3.5))
        x_values = sorted(learned["object_count"].unique())
        x = np.arange(len(x_values))
        bar_w = 0.25
        offsets = [-bar_w / 2, bar_w / 2]
        styles = {
            "Learned-Bloom": {"color": "#d9d9d9", "hatch": "//"},
            "Learned-Cuckoo": {"color": "#969696", "hatch": "xx"},
        }
        for idx, method in enumerate(methods):
            ys = []
            for xv in x_values:
                sub = learned[(learned["object_count"] == xv) & (learned["method"] == method)]
                ys.append(float(sub[m].mean()) if len(sub) else 0.0)
            ax.bar(x + offsets[idx], ys, width=bar_w, label=method, color=styles[method]["color"], hatch=styles[method]["hatch"], edgecolor="black")
        ax.set_xlabel("Number of items (Millions)")
        ax.set_ylabel(lab)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v/1_000_000:.2f}" for v in x_values])
        ax.grid(axis="y")
        ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.24), frameon=True)
        #fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.70))
        fig.subplots_adjust(top=0.62, bottom=0.15, left=0.10, right=0.95)
        out_path = os.path.join(OUTPUT_DIR, f"train_{m}.pdf")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        files.append(out_path)
    return files


def main():
    sim = SimulationV4()
    df = sim.run()
    agg = aggregate_results(df)

    raw_csv = os.path.join(OUTPUT_DIR, "results_raw_v4.csv")
    agg_csv = os.path.join(OUTPUT_DIR, "results_agg_v4.csv")
    df.to_csv(raw_csv, index=False)
    agg.to_csv(agg_csv, index=False)

    latex_path = os.path.join(OUTPUT_DIR, "results_agg_v4.tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(agg.to_latex(index=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else ""))

    plots = []
    plots.append(grouped_bar_plot(agg, "cpu_ops", "CPU ops", "CPU Cost vs Number of Items", "cpu_cost_by_items.pdf"))
    plots.append(grouped_bar_plot(agg, "memory_kb", "Memory (KB)", "Memory Footprint vs Number of Items", "memory_by_items.pdf"))
    plots.append(grouped_bar_plot(agg, "bandwidth_kb", "Bandwidth (KB)", "Bandwidth vs Number of Items", "bandwidth_by_items.pdf"))
    plots.append(grouped_bar_plot(agg, "metadata_ops", "Metadata Ops", "Metadata Operations vs Number of Items", "metadata_ops_by_items.pdf"))
    plots.append(grouped_bar_plot(agg, "dedup_ratio", "Deduplication Ratio", "Deduplication Ratio vs Number of Items", "dedup_ratio_by_items.pdf"))
    plots.append(grouped_bar_plot(agg, "latency_mean_ms", "Latency (ms)", "Latency vs Number of Items", "latency_by_items.pdf"))
    plots.extend(plot_train_test_metrics(agg))

    summary = {
        "raw_csv": raw_csv,
        "agg_csv": agg_csv,
        "latex": latex_path,
        "plots": plots,
        "rows_raw": int(len(df)),
        "rows_agg": int(len(agg)),
    }
    print(pd.Series(summary).to_string())
    print("\nDone. Output directory:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
