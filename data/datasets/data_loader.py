# -*- coding: utf-8 -*-

"""
ReferIt, UNC, UNC+ and GRef referring image segmentation PyTorch dataset.

Define and group batches of images, segmentations and queries.
Based on:
https://github.com/chenxi116/TF-phrasecut-public/blob/master/build_batches.py
"""

import os
import re
# import cv2
import sys
import json
import random
import torch
import numpy as np
import os.path as osp
import scipy.io as sio
import torch.utils.data as data
sys.path.append('.')

from PIL import Image
from pytorch_pretrained_bert.tokenization import BertTokenizer
from utils.word_utils import Corpus
from utils.scene_graph_utils import GraphKeyBuilder

COCO_LIKE_DATASETS = {'unc', 'unc+', 'gref', 'gref_umd'}


def _close_list(a, b, tol: float = 1e-2) -> bool:
    if a is None or b is None:
        return False
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        try:
            xf = float(x)
            yf = float(y)
        except (TypeError, ValueError):
            return False
        if not (np.isfinite(xf) and np.isfinite(yf)):
            return False
        if abs(xf - yf) > tol:
            return False
    return True

def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line #reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples

## Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features

class DatasetNotFoundError(Exception):
    pass

class SegVGDataset(data.Dataset):
    SUPPORTED_DATASETS = {
        'referit': {'splits': ('train', 'val', 'trainval', 'test')},
        'unc': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco', 'split_by': 'unc'}
        },
        'unc+': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
        },
        'gref': {
            'splits': ('train', 'val'),
            'params': {'dataset': 'refcocog', 'split_by': 'google'}
        },
        'gref_umd': {
            'splits': ('train', 'val', 'test'),
            'params': {'dataset': 'refcocog', 'split_by': 'umd'}
        },
        'flickr': {
            'splits': ('train', 'val', 'test')}
    }

    def __init__(self, data_root, split_root='data', dataset='referit', 
                 transform=None, return_idx=False, testmode=False,
                 split='train', max_query_len=128, lstm=False, 
                 bert_model='bert-base-uncased', sgg_config=None):
        self.images = []
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.query_len = max_query_len
        self.lstm = lstm
        self.transform = transform
        self.testmode = testmode
        self.split = split
         
        # Load tokenizer - try local vocab file first for offline mode
        vocab_path = os.path.join(os.path.dirname(__file__), 'bert-base-uncased-vocab.txt')
        if os.path.exists(vocab_path):
            print(f"Loading BertTokenizer from local vocab: {vocab_path}")
            self.tokenizer = BertTokenizer(vocab_path, do_lower_case=True)
        else:
            print(f"Loading BertTokenizer from pretrained: {bert_model}")
            self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)

        self.return_idx = return_idx
        self.sgg_config = sgg_config or {}
        self.sgg_enabled = bool(self.sgg_config.get('enabled', False))
        self.sgg_num_edges = self.sgg_config.get('num_edges', 3)
        self.sgg_relation_max = self.sgg_config.get('relation_max', 32)
        self.sgg_graph_root = None
        self.sgg_token_max_len = min(self.query_len, self.sgg_config.get('token_max_len', 32))
        self.sgg_aug_prob = self.sgg_config.get('aug_prob', 0.0)
        self._sgg_augment_enabled = (self.sgg_enabled and self.split == 'train'
                                     and self.sgg_aug_prob > 0)
        self._eval_augmented_query = bool(self.sgg_config.get('eval_augmented_query', False))

        assert self.transform is not None

        if split == 'train':
            self.augment = True
        else:
            self.augment = False

        if self.dataset == 'referit':
            self.dataset_root = osp.join(self.data_root, 'referit')
            self.im_dir = osp.join(self.dataset_root, 'images')
            self.split_dir = osp.join(self.dataset_root, 'splits')
        elif  self.dataset == 'flickr':
            self.dataset_root = osp.join(self.data_root, 'Flickr30k')
            self.im_dir = osp.join(self.dataset_root, 'flickr30k_images')
        else:   ## refcoco, etc.
            self.dataset_root = osp.join(self.data_root, 'other')
            self.im_dir = osp.join(
                self.dataset_root, 'images', 'mscoco', 'images', 'train2014')

        if not self.exists_dataset():
            # self.process_dataset()
            test_p = osp.join(self.split_root, self.dataset)
            print(f'--- not found dataset: {test_p}')
            print('Please download index cache to data folder: \n \
                https://drive.google.com/open?id=1cZI562MABLtAzM6YU4WmKPFFguuVr0lZ')
            exit(0)

        dataset_path = osp.join(self.split_root, self.dataset)
        valid_splits = self.SUPPORTED_DATASETS[self.dataset]['splits']

        if self.lstm:
            self.corpus = Corpus()
            corpus_path = osp.join(dataset_path, 'corpus.pth')
            self.corpus = torch.load(corpus_path)

        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self.dataset, split))

        splits = [split]
        if self.dataset != 'referit':
            splits = ['train', 'val'] if split == 'trainval' else [split]
        for split in splits:
            imgset_file = '{0}_{1}.pth'.format(self.dataset, split)
            imgset_path = osp.join(dataset_path, imgset_file)
            self.images += torch.load(imgset_path)

        self._init_graph_keys()
        self._init_sgg_support()
        self._load_graph_caches()
        
    def exists_dataset(self):
        return osp.exists(osp.join(self.split_root, self.dataset))

    def pull_item(self, idx):
        if self.dataset == 'flickr':
            img_file, bbox, phrase = self.images[idx]
            mask_file = ''
        else:
            img_file, mask_file, bbox, phrase, attri = self.images[idx]
        ## box format: to x1y1x2y2
        if not (self.dataset == 'referit' or self.dataset == 'flickr'):
            bbox = np.array(bbox, dtype=int)
            bbox[2], bbox[3] = bbox[0]+bbox[2], bbox[1]+bbox[3]
        else:
            bbox = np.array(bbox, dtype=int)

        img_path = osp.join(self.im_dir, img_file)
        img = Image.open(img_path).convert("RGB")

        bbox = torch.tensor(bbox)
        bbox = bbox.float()
        return img, phrase, bbox, mask_file

    def tokenize_phrase(self, phrase):
        return self.corpus.tokenize(phrase, self.query_len)

    def untokenize_word_vector(self, words):
        return self.corpus.dictionary[words]

    def __len__(self):
        base_len = len(self.images)
        if self._sgg_augment_enabled:
            return base_len * (1 + self.sgg_num_edges)
        return base_len

    def _init_graph_keys(self):
        prefer_mask_name = self.dataset in COCO_LIKE_DATASETS
        builder = GraphKeyBuilder(prefer_mask_name=prefer_mask_name)
        self.graph_keys = []
        self.graph_occurrences = []
        for idx, entry in enumerate(self.images):
            if self.dataset == 'flickr':
                img_file, _, phrase = entry
                mask_name = None
            else:
                img_file, mask_file, _, phrase, *_ = entry
                mask_name = osp.splitext(mask_file)[0] if mask_file else None
            graph_key, occurrence = builder.build(idx, img_file, phrase, mask_name)
            self.graph_keys.append(graph_key)
            self.graph_occurrences.append(occurrence)

    def __getitem__(self, idx):
        # --- SGG augmentation: resolve original index and augmentation slot ---
        aug_slot = 0
        orig_idx = idx
        if self._sgg_augment_enabled:
            n = 1 + self.sgg_num_edges  # typically 4
            orig_idx = idx // n
            aug_slot = idx % n  # 0 = original, 1-3 = edge augmentation slots

        record = self.images[orig_idx]
        if self.dataset == 'flickr':
            img_file, _, raw_phrase = record
            mask_file = ''
            raw_bbox_for_metadata = None
        else:
            img_file, mask_file, _, raw_phrase, *_ = record
            raw_bbox_for_metadata = np.array(record[2], dtype=float) if len(record) > 2 else None

        graph_key = self.graph_keys[orig_idx] if self.sgg_enabled else None
        graph_occurrence = self.graph_occurrences[orig_idx] if self.sgg_enabled else None

        img, phrase, bbox, mask = self.pull_item(orig_idx)
        raw_phrase_for_metadata = phrase
        bbox_for_metadata = bbox.clone()

        # Load SGG target early (needed for phrase augmentation)
        sgg_target = None
        if self.sgg_enabled and mask:
            sgg_target = self._load_sgg_target(
                orig_idx,
                mask,
                graph_key,
                graph_occurrence,
                img_file=img_file,
                phrase=raw_phrase_for_metadata,
                bbox=bbox_for_metadata,
                raw_bbox=raw_bbox_for_metadata,
            )

        # --- SGG phrase augmentation ---
        # aug_slot > 0 means this index is an augmentation slot for edge (aug_slot-1).
        # With sgg_aug_prob probability, append "<relation> <attrs> <object>" to the
        # grounding phrase.  Otherwise fall through with the original phrase.
        if aug_slot > 0 and sgg_target is not None:
            edge_idx = aug_slot - 1
            relations = sgg_target.get('relations', [])
            if (edge_idx < len(relations)
                    and relations[edge_idx].get('valid', False)
                    and random.random() < self.sgg_aug_prob):
                phrase = self._build_augmented_phrase(phrase, relations[edge_idx])

        # --- Eval with augmented query (overfitting analysis) ---
        # Replace the original query with the simplest valid augmented query
        # from the scene graph edges. "Simplest" = shortest appended text.
        if self._eval_augmented_query and sgg_target is not None:
            phrase = self._pick_simplest_augmented_phrase(phrase, sgg_target)

        phrase = phrase.lower()
        input_dict = {'img': img, 'box': bbox, 'text': phrase}
        input_dict = self.transform(input_dict)
        img = input_dict['img']
        bbox = input_dict['box']
        phrase = input_dict['text']
        img_mask = input_dict['mask']
        pos_region = input_dict['pos_region']
        
        if self.lstm:
            phrase = self.tokenize_phrase(phrase)
            word_id = phrase
            word_mask = np.array(word_id>0, dtype=int)
        else:
            ## encode phrase to bert input
            examples = read_examples(phrase, orig_idx)
            features = convert_examples_to_features(
                examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
            word_id = features[0].input_ids
            word_mask = features[0].input_mask

        return img, np.array(img_mask), np.array(word_id, dtype=int), np.array(word_mask, dtype=int), np.array(bbox, dtype=np.float32), np.array(pos_region), sgg_target

    def _build_augmented_phrase(self, phrase, edge):
        """Build augmented grounding phrase: '<phrase> <relation> <attributes> <object>'.

        Example: 'man second from right' + edge{holding, [red], cup}
              -> 'man second from right holding red cup'
        """
        relation = edge.get('relation_name', '').strip()
        if not relation or relation == 'none':
            return phrase
        obj_attrs = edge.get('object_attributes', [])
        obj_name = edge.get('object_name', '').strip()
        # Build object description: "red cup", "orange ball"
        parts = [a.strip() for a in obj_attrs if a.strip()]
        if obj_name and obj_name != 'none':
            parts.append(obj_name)
        obj_desc = ' '.join(parts)
        if obj_desc:
            return f"{phrase} {relation} {obj_desc}"
        return f"{phrase} {relation}"

    def _pick_simplest_augmented_phrase(self, phrase, sgg_target):
        """Replace original phrase with the simplest augmented query from scene graph edges.

        Iterates over the (up to 3) relation edges, builds the augmented suffix
        for each valid edge, and picks the one with the shortest suffix text.
        If no valid edge exists the original phrase is returned unchanged.
        """
        relations = sgg_target.get('relations', [])
        best_phrase = None
        best_suffix_len = float('inf')
        for edge in relations:
            if not edge.get('valid', False):
                continue
            relation = edge.get('relation_name', '').strip()
            if not relation or relation == 'none':
                continue
            # Build the suffix that would be appended
            obj_attrs = edge.get('object_attributes', [])
            obj_name = edge.get('object_name', '').strip()
            parts = [a.strip() for a in obj_attrs if a.strip()]
            if obj_name and obj_name != 'none':
                parts.append(obj_name)
            obj_desc = ' '.join(parts)
            suffix = f"{relation} {obj_desc}".strip() if obj_desc else relation
            if len(suffix) < best_suffix_len:
                best_suffix_len = len(suffix)
                best_phrase = f"{phrase} {suffix}"
        return best_phrase if best_phrase is not None else phrase

    def _tokenize_texts(self, texts):
        if not texts:
            texts = ['none']
        tokenizer_callable = callable(getattr(self.tokenizer, '__call__', None))
        if tokenizer_callable:
            encoding = self.tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=self.sgg_token_max_len,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].cpu().numpy().astype(np.int64)
            attention_mask = encoding['attention_mask'].cpu().numpy().astype(np.int64)
            return input_ids, attention_mask

        max_len = self.sgg_token_max_len
        input_id_list = []
        attention_list = []
        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > max_len - 2:
                tokens = tokens[:max_len - 2]
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            attention = [1] * len(token_ids)
            pad_len = max_len - len(token_ids)
            if pad_len > 0:
                token_ids += [0] * pad_len
                attention += [0] * pad_len
            elif pad_len < 0:
                token_ids = token_ids[:max_len]
                attention = attention[:max_len]
            input_id_list.append(token_ids)
            attention_list.append(attention)

        input_ids = np.array(input_id_list, dtype=np.int64)
        attention_mask = np.array(attention_list, dtype=np.int64)
        return input_ids, attention_mask

    def _init_sgg_support(self):
        enabled = bool(self.sgg_config.get('enabled', False))
        graph_root = self.sgg_config.get('annotation_root')
        if not enabled or not graph_root:
            return

        graph_root = osp.expanduser(graph_root)
        if not osp.isabs(graph_root):
            graph_root = osp.abspath(graph_root)
        if not osp.isdir(graph_root):
            return

        self.sgg_enabled = True
        self.sgg_graph_root = graph_root
        
        # Will be populated by _load_graph_caches
        self._graph_cache = {}

        if self.split == 'train':
            # Note: Filtering will now use _graph_exists instead of _resolve_graph_path
            # This is done in _load_graph_caches after caches are loaded
            pass
    
    def _load_graph_caches(self):
        """Load consolidated .pth files into memory for fast access."""
        if not self.sgg_enabled or not self.sgg_graph_root:
            return
        
        # Determine which .pth files to load
        is_coco_like = self.dataset in COCO_LIKE_DATASETS
        
        # Get actual splits (for trainval, we need both train and val)
        if self.split == 'trainval':
            splits_to_load = ['train', 'val']
        else:
            splits_to_load = [self.split]
        
        # Load split-specific graphs
        for split in splits_to_load:
            pth_file = osp.join(self.sgg_graph_root, f"{self.dataset}_graphs_{split}.pth")
            if osp.exists(pth_file):
                print(f"Loading scene graphs from {pth_file}...")
                try:
                    graphs = torch.load(pth_file)
                    self._graph_cache.update(graphs)
                    print(f"  Loaded {len(graphs)} graphs from {split} split")
                except Exception as e:
                    print(f"  Warning: Failed to load {pth_file}: {e}")
        
        # Load shared graphs for COCO-like datasets
        if is_coco_like:
            shared_pth = osp.join(self.sgg_graph_root, f"{self.dataset}_graphs_shared.pth")
            if osp.exists(shared_pth):
                print(f"Loading shared scene graphs from {shared_pth}...")
                try:
                    shared_graphs = torch.load(shared_pth)
                    # Shared graphs have lower priority - don't overwrite split-specific ones
                    for key, value in shared_graphs.items():
                        if key not in self._graph_cache:
                            self._graph_cache[key] = value
                    print(f"  Loaded {len(shared_graphs)} shared graphs")
                except Exception as e:
                    print(f"  Warning: Failed to load {shared_pth}: {e}")
        
        print(f"Total graphs loaded in cache: {len(self._graph_cache)}")

    def _graph_exists(self, graph_key=None, mask_name=None):
        """Check if a graph exists in the loaded cache."""
        if not hasattr(self, '_graph_cache'):
            return False
        
        # Try graph_key first, then mask_name
        if graph_key and graph_key in self._graph_cache:
            return True
        if mask_name and mask_name in self._graph_cache:
            return True
        return False
    
    def _get_graph_from_cache(self, graph_key=None, mask_name=None):
        """Retrieve graph from cache. Returns (graph_data, source_key)."""
        if not hasattr(self, '_graph_cache'):
            return None, None
        
        # Try graph_key first (higher priority)
        if graph_key and graph_key in self._graph_cache:
            return self._graph_cache[graph_key], graph_key
        
        # Fall back to mask_name
        if mask_name and mask_name in self._graph_cache:
            return self._graph_cache[mask_name], mask_name
        
        return None, None

    def _graph_search_dirs(self):
        """Legacy method - kept for compatibility but not used with .pth caches."""
        dirs = []
        if not self.sgg_graph_root:
            return dirs
        split_dir = osp.join(self.sgg_graph_root, self.split)
        shared_dir = osp.join(self.sgg_graph_root, 'shared')
        if osp.isdir(split_dir):
            dirs.append(split_dir)
        allow_shared = self.dataset not in COCO_LIKE_DATASETS
        if allow_shared and osp.isdir(shared_dir) and shared_dir not in dirs:
            dirs.append(shared_dir)
        if allow_shared and osp.isdir(self.sgg_graph_root):
            dirs.append(self.sgg_graph_root)
        return dirs

    def _graph_candidates(self, graph_key=None, mask_name=None):
        """Legacy method - kept for compatibility but not used with .pth caches."""
        candidates = []
        for base_dir in self._graph_search_dirs():
            if graph_key:
                candidates.append((osp.join(base_dir, f'{graph_key}.json'), 'graph_key'))
            if mask_name:
                candidates.append((osp.join(base_dir, f'{mask_name}.json'), 'mask'))
        return candidates

    def _resolve_graph_path(self, graph_key=None, mask_name=None):
        """
        Legacy method for JSON file lookup - kept for backward compatibility.
        Now returns (None, 'cache') to indicate graph should be loaded from cache.
        Falls back to JSON file search if cache is not available.
        """
        # If we have cache loaded, indicate that graph should come from cache
        if hasattr(self, '_graph_cache') and self._graph_cache:
            if self._graph_exists(graph_key, mask_name):
                return None, 'cache'  # Signal to use cache instead of file path
            else:
                return None, None  # Graph not found
        
        # Fallback to old JSON file search (for backward compatibility if .pth files don't exist)
        if not self.sgg_graph_root:
            return None, None
        for path, source in self._graph_candidates(graph_key, mask_name):
            if osp.isfile(path):
                return path, source
        return None, None

    def _load_sgg_target(self, idx, mask_file, graph_key=None, graph_occurrence=None,
                         img_file=None, phrase=None, bbox=None, raw_bbox=None):
        mask_name = osp.splitext(mask_file)[0]
        
        # Try to load from cache first
        graph, source_key = self._get_graph_from_cache(graph_key, mask_name)
        
        if graph is None:
            # Fallback to loading from JSON file (backward compatibility)
            graph_path, source = self._resolve_graph_path(graph_key, mask_name)
            if not graph_path:
                # Return None instead of raising error to allow training without SGG
                return None
            
            try:
                with open(graph_path, 'r') as f:
                    graph = json.load(f)
                source = 'file'
            except Exception as e:
                # Return None if file cannot be read
                return None
        else:
            # Graph loaded from cache
            source = 'cache'

        nodes = {node['id']: node for node in graph.get('nodes', [])}
        edges = graph.get('edges', [])
        # Note: metadata is not validated as it's not considered authoritative
        # The graph key lookup is sufficient to ensure correct graph retrieval

        processed_rel = []
        for edge in edges:
            obj_id = edge.get('object_id')
            obj_node = nodes.get(obj_id, {})
            obj_name = obj_node.get('name', 'object')
            obj_attrs = obj_node.get('attributes', [])
            relation = edge.get('relation_name', 'none')
            relation = relation.lower().strip()
            confidence = float(edge.get('confidence', 1.0))
            if confidence <= 0.0:
                confidence = 1.0
            description_tokens = obj_attrs + ([obj_name] if obj_name else [])
            processed_rel.append({
                'relation_name': relation if relation else 'none',
                'object_name': obj_name,
                'object_attributes': obj_attrs,
                'object_description': ' '.join(description_tokens).strip() if description_tokens else obj_name,
                'confidence': confidence,
                'object_id': obj_id,
                'raw_edge': edge,
                'object_assignment': None,
                'valid': True,
            })

        # Sort by confidence descending to prioritise higher quality annotations
        processed_rel.sort(key=lambda x: x['confidence'], reverse=True)

        fixed_relations = []
        valid_mask = []
        for idx in range(self.sgg_num_edges):
            if idx < len(processed_rel):
                rel = processed_rel[idx]
                fixed_relations.append(rel)
                valid_mask.append(True)
            else:
                fixed_relations.append({
                    'relation_name': 'none',
                    'object_name': 'none',
                    'object_attributes': [],
                    'object_description': 'none',
                    'confidence': 0.0,
                    'object_id': 0,
                    'raw_edge': None,
                    'object_assignment': None,
                    'valid': False,
                })
                valid_mask.append(False)

        relation_strings = []
        for rel in fixed_relations:
            name = rel['relation_name']
            if name == 'none':
                continue
            if name not in relation_strings:
                relation_strings.append(name)
            if len(relation_strings) >= self.sgg_relation_max:
                break

        if 'none' not in relation_strings:
            if len(relation_strings) < self.sgg_relation_max:
                relation_strings.append('none')
            else:
                relation_strings[-1] = 'none'
        relation_strings = relation_strings[:self.sgg_relation_max]

        relation_input_ids, relation_attention = self._tokenize_texts(relation_strings)

        relation_index_map = {name: idx for idx, name in enumerate(relation_strings)}
        edge_relation_indices = [relation_index_map.get(rel['relation_name'], -1) for rel in fixed_relations]

        object_descriptions = [rel['object_description'] if rel['object_description'] else 'none' for rel in fixed_relations]
        object_input_ids, object_attention = self._tokenize_texts(object_descriptions)

        confidence = np.array([rel['confidence'] for rel in fixed_relations], dtype=np.float32)
        valid_mask_np = np.array(valid_mask, dtype=np.bool_)
        confidence = np.where((confidence <= 0.0) & valid_mask_np, 1.0, confidence)
        edge_relation_indices = np.array(edge_relation_indices, dtype=np.int64)

        # Create minimal metadata (graph metadata is not considered authoritative)
        metadata = {
            'graph_key': graph_key,
            'graph_occurrence': graph_occurrence,
            'dataset_index': idx,
            'graph_source': source,
        }

        return {
            'graph_path': source_key if source == 'cache' else graph_path,
            'mask_name': mask_name,
            'image_id': img_file,
            'metadata': metadata,
            'relations': fixed_relations,
            'relation_strings': relation_strings,
            'relation_token_ids': relation_input_ids,
            'relation_token_masks': relation_attention,
            'object_token_ids': object_input_ids,
            'object_token_masks': object_attention,
            'edge_relation_indices': edge_relation_indices,
            'valid_mask': valid_mask_np,
            'confidence': confidence,
            'nodes': nodes,
        }