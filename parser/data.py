import logging
import re
import json
import random
import torch
from torch import nn
import numpy as np
from parser.AMRGraph import AMRGraph
from parser.extract import read_file

logger=logging

PAD, UNK, DUM, NIL, END, CLS = '<PAD>', '<UNK>', '<DUMMY>', '<NULL>', '<END>', '<CLS>'
GPU_SIZE = 12000 # okay for 8G memory

class Vocab(object):
    def __init__(self, filename, min_occur_cnt, specials = None):
        idx2token = [PAD, UNK] + (specials if specials is not None else [])
        self._priority = dict()
        num_tot_tokens = 0
        num_vocab_tokens = 0
        for line in open(filename).readlines():
            try:
                token, cnt = line.rstrip('\n').split('\t')
                cnt = int(cnt)
                num_tot_tokens += cnt
            except:
                print(line)
            if cnt >= min_occur_cnt:
                idx2token.append(token)
                num_vocab_tokens += cnt
            self._priority[token] = int(cnt)
        self.coverage = num_vocab_tokens/num_tot_tokens
        self._token2idx = dict(zip(idx2token, range(len(idx2token))))
        self._idx2token = idx2token
        self._padding_idx = self._token2idx[PAD]
        self._unk_idx = self._token2idx[UNK]
    
    def priority(self, x):
        return self._priority.get(x, 0)
    
    @property
    def size(self):
        return len(self._idx2token)

    @property
    def unk_idx(self):
        return self._unk_idx

    @property
    def padding_idx(self):
        return self._padding_idx

    def idx2token(self, x):
        if isinstance(x, list):
            return [self.idx2token(i) for i in x]
        return self._idx2token[x]

    def token2idx(self, x):
        if isinstance(x, list):
            return [self.token2idx(i) for i in x]
        return self._token2idx.get(x, self.unk_idx)

def ListsToTensor(xs, vocab=None, local_vocabs=None, unk_rate=0.):
    pad = vocab.padding_idx if vocab else 0
    
    def toIdx(w, i):
        if vocab is None:
            return w
        if isinstance(w, list):
            return [toIdx(_, i) for _ in w]
        if random.random() < unk_rate:
            return vocab.unk_idx
        if local_vocabs is not None:
            local_vocab = local_vocabs[i]
            if (local_vocab is not None) and (w in local_vocab):
                return local_vocab[w]
        return vocab.token2idx(w)

    max_len = max(len(x) for x in xs)
    ys = []
    for i, x in enumerate(xs):
        y = toIdx(x, i) + [pad]*(max_len-len(x))
        ys.append(y)
    data = np.transpose(np.array(ys))
    return data

def ListsofStringToTensor(xs, vocab, max_string_len=20):
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs:
        y = x + [PAD]*(max_len -len(x))
        zs = []
        for z in y:
            z = list(z[:max_string_len])
            zs.append(vocab.token2idx([CLS]+z+[END]) + [vocab.padding_idx]*(max_string_len - len(z)))
        ys.append(zs)

    data = np.transpose(np.array(ys), (1, 0, 2))
    return data

def ArraysToTensor(xs):
    "list of numpy array, each has the same demonsionality"
    x = np.array([ list(x.shape) for x in xs])
    shape = [len(xs)] + list(x.max(axis = 0))
    data = np.zeros(shape, dtype=np.int)
    for i, x in enumerate(xs):
        slicing_shape = list(x.shape)
        slices = tuple([slice(i, i+1)]+[slice(0, x) for x in slicing_shape])
        data[slices] = x
    #tensor = torch.from_numpy(data).long()
    return data

def batchify(data, vocabs, unk_rate=0.):
    _tok = ListsToTensor([ [CLS]+x['tok'] for x in data], vocabs['tok'], unk_rate=unk_rate)
    _lem = ListsToTensor([ [CLS]+x['lem'] for x in data], vocabs['lem'], unk_rate=unk_rate)
    _pos = ListsToTensor([ [CLS]+x['pos'] for x in data], vocabs['pos'], unk_rate=unk_rate)
    _ner = ListsToTensor([ [CLS]+x['ner'] for x in data], vocabs['ner'], unk_rate=unk_rate)
    _word_char = ListsofStringToTensor([ [CLS]+x['tok'] for x in data], vocabs['word_char'])

    local_token2idx = [x['token2idx'] for x in data]
    local_idx2token = [x['idx2token'] for x in data]
    _cp_seq = ListsToTensor([ x['cp_seq'] for x in data], vocabs['predictable_concept'], local_token2idx)
    _mp_seq = ListsToTensor([ x['mp_seq'] for x in data], vocabs['predictable_concept'], local_token2idx)

    concept, edge = [], []
    for x in data:
        amr = x['amr']
        concept_i, edge_i, _ = amr.root_centered_sort(vocabs['rel'].priority)
        concept.append(concept_i)
        edge.append(edge_i)

    augmented_concept = [[DUM]+x+[END] for x in concept]

    _concept_in = ListsToTensor(augmented_concept, vocabs['concept'], unk_rate=unk_rate)[:-1]
    _concept_char_in = ListsofStringToTensor(augmented_concept, vocabs['concept_char'])[:-1]
    _concept_out = ListsToTensor(augmented_concept, vocabs['predictable_concept'], local_token2idx)[1:]

    out_conc_len, bsz = _concept_out.shape
    _rel = np.full((1+out_conc_len, bsz, out_conc_len), vocabs['rel'].token2idx(PAD))
    # v: [<dummy>, concept_0, ..., concept_l, ..., concept_{n-1}, <end>] u: [<dummy>, concept_0, ..., concept_l, ..., concept_{n-1}]
    
    for bidx, (x, y) in enumerate(zip(edge, concept)):
        for l, _ in enumerate(y):
            if l > 0:
                # l=1 => pos=l+1=2
                _rel[l+1, bidx, 1:l+1] = vocabs['rel'].token2idx(NIL)
        for v, u, r in x:
            r = vocabs['rel'].token2idx(r)
            _rel[v+1, bidx, u+1] = r

    ret = {'lem':_lem, 'tok':_tok, 'pos':_pos, 'ner':_ner, 'word_char':_word_char, \
           'copy_seq': np.stack([_cp_seq, _mp_seq], -1), \
           'local_token2idx':local_token2idx, 'local_idx2token': local_idx2token, \
           'concept_in':_concept_in, 'concept_char_in':_concept_char_in, \
           'concept_out':_concept_out, 'rel':_rel}

    bert_tokenizer = vocabs.get('bert_tokenizer', None) 
    if bert_tokenizer is not None:
        ret['bert_token'] = ArraysToTensor([ x['bert_token'] for x in data])
        ret['token_subword_index'] = ArraysToTensor([ x['token_subword_index'] for x in data])
    return ret
    
class DataLoader(object):
    def __init__(self, vocabs, lex_map, filename, batch_size, for_train, **kwargs):
        self.data = []
        bert_tokenizer = vocabs.get('bert_tokenizer', None)
        for amr, token, lemma, pos, ner in zip(*read_file(filename)):
            if for_train:
                _, _, not_ok = amr.root_centered_sort()
                if not_ok or len(token)==0:
                    continue
            cp_seq, mp_seq, token2idx, idx2token = lex_map.get_concepts(lemma, token, vocabs['predictable_concept']) 
            datum = {'amr':amr, 'tok':token, 'lem':lemma, 'pos':pos, 'ner':ner, \
                     'cp_seq':cp_seq, 'mp_seq':mp_seq,\
                     'token2idx':token2idx, 'idx2token':idx2token}
            if bert_tokenizer is not None:
                bert_token, token_subword_index = bert_tokenizer.tokenize(token)
                datum['bert_token'] = bert_token
                datum['token_subword_index'] = token_subword_index


            self.data.append(datum)
        print ("Get %d AMRs from %s"%(len(self.data), filename))
        self.vocabs = vocabs
        self.batch_size = batch_size
        self.train = for_train
        self.unk_rate = 0.

    def set_unk_rate(self, x):
        self.unk_rate = x

    def __iter__(self):
        idx = list(range(len(self.data)))
        
        if self.train:
            random.shuffle(idx)
            idx.sort(key = lambda x: len(self.data[x]['tok']) + len(self.data[x]['amr']))

        batches = []
        num_tokens, data = 0, []
        for i in idx:
            num_tokens += len(self.data[i]['tok']) + len(self.data[i]['amr'])
            data.append(self.data[i])
            if num_tokens >= self.batch_size:
                sz = len(data)* (2 + max(len(x['tok']) for x in data) + max(len(x['amr']) for x in data))
                if sz > GPU_SIZE:
                    # because we only have limited GPU memory
                    batches.append(data[:len(data)//2])
                    data = data[len(data)//2:]
                batches.append(data)
                num_tokens, data = 0, []
        if data:
            sz = len(data)* (2 + max(len(x['tok']) for x in data) + max(len(x['amr']) for x in data))
            if sz > GPU_SIZE:
                # because we only have limited GPU memory
                batches.append(data[:len(data)//2])
                data = data[len(data)//2:]
            batches.append(data)

        if self.train:
            random.shuffle(batches)

        for batch in batches:
            yield batchify(batch, self.vocabs, self.unk_rate)


class Curriculum:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_difficulty = None

    def compute_data_difficulty(self):
        raw = np.array([
            self.get_item_difficulty(item) for item in self.data
        ],dtype='float')
        cdf = (raw[None,:] <= raw[:,None]).mean(axis=1)
        self.data_difficulty = cdf
        self.raw_data_difficulty = raw
    
    def difficulty(self, itemidx):
        if self.data_difficulty is None: 
            self.compute_data_difficulty()
        return self.data_difficulty[itemidx]

class MeanCurriculum(Curriculum):
    def compute_data_difficulty(self):
        raw = np.array([
            self.get_item_difficulty(item) for item in self.data
        ],dtype='float')

        # the further from the mean the more difficult
        dist2mean = np.abs(raw - raw.mean())

        cdf = (dist2mean[None,:] <= dist2mean[:,None]).mean(axis=1)
        self.data_difficulty = cdf
        self.raw_data_difficulty = dist2mean
    

class CompetenceBasedCurriculumDataIter:
    def __init__(self, *args, curriculum_len=None, initial_competence=None, slope_power=2, **kwargs):
        assert curriculum_len
        assert initial_competence
        assert float(slope_power)
        
        super().__init__(*args, **kwargs)

        self.curriculum_len = curriculum_len
        self.initial_competence = initial_competence
        self.initial_competence_powered = initial_competence ** slope_power
        self.slope_power = slope_power
        self.timestep = -1
    
    def difficulty(self, itemidx): 
        return super().difficulty(itemidx)

    def competence(self, timestep):
        return 1 if timestep >= self.curriculum_len else ( timestep * ( 1 - self.initial_competence_powered ) / self.curriculum_len + self.initial_competence_powered ) ** ( 1 / self.slope_power ) 

    def __iter__(self):
        if not self.train: 
            yield from super().__iter__()
            return

        self.timestep += 1
        timestep = self.timestep

        current_competence = self.competence(timestep)

        sampled_data = [item for idx, item in enumerate(self.data) if self.difficulty(idx) <= current_competence + 0.0001]
        idx = list(range(len(sampled_data)))
        random.shuffle(idx)
        idx.sort(key = lambda x: len(sampled_data[x]['tok']) + len(sampled_data[x]['amr']))

        batches = []
        num_tokens, data = 0, []
        for i in idx:
            num_tokens += len(sampled_data[i]['tok']) + len(sampled_data[i]['amr'])
            data.append(sampled_data[i])
            if num_tokens >= self.batch_size:
                sz = len(data)* (2 + max(len(x['tok']) for x in data) + max(len(x['amr']) for x in data))
                if sz > GPU_SIZE:
                    # because we only have limited GPU memory
                    batches.append(data[:len(data)//2])
                    data = data[len(data)//2:]
                batches.append(data)
                num_tokens, data = 0, []
        if data:
            sz = len(data)* (2 + max(len(x['tok']) for x in data) + max(len(x['amr']) for x in data))
            if sz > GPU_SIZE:
                # because we only have limited GPU memory
                batches.append(data[:len(data)//2])
                data = data[len(data)//2:]
            batches.append(data)
        
        random.shuffle(batches)
        
        logger.info('timestep %d, sample-size %d, batches %d, competence %.3f'%(timestep, len(sampled_data), len(batches), current_competence))
        for batch in batches:
            yield batchify(batch, self.vocabs, self.unk_rate)

class BatchCompetenceBasedCurriculumDataIter:
    def __init__(self, *args, curriculum_len=None, initial_competence=None, slope_power=2, **kwargs):
        assert curriculum_len
        assert initial_competence
        assert float(slope_power)
        
        super().__init__(*args, **kwargs)

        self.curriculum_len = curriculum_len
        self.initial_competence = initial_competence
        self.initial_competence_powered = initial_competence ** slope_power
        self.slope_power = slope_power
        self.timestep = -1
    
    def difficulty(self, itemidx): 
        return super().difficulty(itemidx)

    def competence(self, timestep):
        return 1 if timestep >= self.curriculum_len else ( timestep * ( 1 - self.initial_competence_powered ) / self.curriculum_len + self.initial_competence_powered ) ** ( 1 / self.slope_power ) 

    def __iter__(self):
        if not self.train: 
            yield from super().__iter__()
            return

        if self.timestep >= self.curriculum_len:
            yield from super().__iter__()
            return

        curriculum_stat={'nexamples':0,'timestep':0,'competence':0}
        for timestep in range(self.curriculum_len):
            self.timestep = timestep
            current_competence = self.competence(timestep)
            sampled_data = [item for idx, item in enumerate(self.data) if self.difficulty(idx) <= current_competence + 0.0001]
            idx = list(range(len(sampled_data)))
            random.shuffle(idx)
            num_tokens, data = 0, []
            
            for i in idx:
                num_tokens += len(sampled_data[i]['tok']) + len(sampled_data[i]['amr'])
                data.append(sampled_data[i])
                if num_tokens >= self.batch_size:
                    break
            if data:
                sz = len(data)* (2 + max(len(x['tok']) for x in data) + max(len(x['amr']) for x in data))
                if sz > GPU_SIZE:
                    # because we only have limited GPU memory
                    batch = data[:len(data)//2]
                else:
                    batch = data
                yield batchify(batch, self.vocabs, self.unk_rate)

            curriculum_stat['nexamples'] += len(batch)
            curriculum_stat['timestep'] = timestep
            curriculum_stat['competence'] = current_competence
            if (timestep + 1) % int(self.curriculum_len/10) == 0:
                logger.info('curriculum-stat %s'%json.dumps(curriculum_stat))

        self.timestep += 1

class OrderedCurriculumDataIter:

    def difficulty(self, itemidx): 
        return super().difficulty(itemidx)

    def __iter__(self):
        if not self.train: 
            yield from super().__iter__()
            return

        sampled_data = self.data
        idx = list(range(len(sampled_data)))
        idx.sort(key=self.difficulty)

        batches = []
        num_tokens, data = 0, []
        for i in idx:
            num_tokens += len(sampled_data[i]['tok']) + len(sampled_data[i]['amr'])
            data.append(sampled_data[i])
            if num_tokens >= self.batch_size:
                sz = len(data)* (2 + max(len(x['tok']) for x in data) + max(len(x['amr']) for x in data))
                if sz > GPU_SIZE:
                    # because we only have limited GPU memory
                    batches.append(data[:len(data)//2])
                    data = data[len(data)//2:]
                batches.append(data)
                num_tokens, data = 0, []
        if data:
            sz = len(data)* (2 + max(len(x['tok']) for x in data) + max(len(x['amr']) for x in data))
            if sz > GPU_SIZE:
                # because we only have limited GPU memory
                batches.append(data[:len(data)//2])
                data = data[len(data)//2:]
            batches.append(data)

        for batch in batches:
            yield batchify(batch, self.vocabs, self.unk_rate)


class WaveCurriculumDataIter:

    def difficulty(self, itemidx): 
        return super().difficulty(itemidx)

    def __iter__(self):
        if not self.train: 
            yield from super().__iter__()
            return

        sampled_data = self.data
        idx = list(range(len(sampled_data)))
        idx.sort(key=self.difficulty)
        idx = idx[::2] + idx[1::2][::-1]

        batches = []
        num_tokens, data = 0, []
        for i in idx:
            num_tokens += len(sampled_data[i]['tok']) + len(sampled_data[i]['amr'])
            data.append(sampled_data[i])
            if num_tokens >= self.batch_size:
                sz = len(data)* (2 + max(len(x['tok']) for x in data) + max(len(x['amr']) for x in data))
                if sz > GPU_SIZE:
                    # because we only have limited GPU memory
                    batches.append(data[:len(data)//2])
                    data = data[len(data)//2:]
                batches.append(data)
                num_tokens, data = 0, []
        if data:
            sz = len(data)* (2 + max(len(x['tok']) for x in data) + max(len(x['amr']) for x in data))
            if sz > GPU_SIZE:
                # because we only have limited GPU memory
                batches.append(data[:len(data)//2])
                data = data[len(data)//2:]
            batches.append(data)

        for batch in batches:
            yield batchify(batch, self.vocabs, self.unk_rate)


class DAMR:

    def get_amr_difficulty(self, amr): raise NotImplementedError

    def get_item_difficulty(self, item): 
        return self.get_amr_difficulty(item['amr'])
    
    def __init__(self,*args,**kwargs):
        self.concept_idf=None
        self.rel_idf=None
        super().__init__(*args,**kwargs)

    def compute_concept_idf(self):
        df = {}
        for item in self.data:
            for node in item['amr'].nodes:
                concept = item['amr'].name2concept[node]
                df[concept] = df.get(concept,0)+1
        self.concept_idf = {c:np.log(len(self.data)/v) for c,v in df.items()}

    def compute_rel_idf(self):
        df = {}
        for item in self.data:
            for src,tgts in item['amr'].edges.items():
                for rel,tgt in tgts:
                    df[rel] = df.get(rel, 0) + 1
        self.rel_idf = {c:np.log(len(self.data)/v) for c,v in df.items()}

    def rel_difficulty(self,rel):
        if self.rel_idf is None:
            self.compute_rel_idf()
        if 'ARG' in rel: return 1
        return 1 + self.rel_idf.get(rel, 1) 
    
    def concept_difficulty(self, concept):
        if self.concept_idf is None:
            self.compute_concept_idf()
        return 1 + self.concept_idf.get(concept, 1)

    
class DAMRR0(DAMR):
    def rel_difficulty(self,rel):
#         if self.rel_idf is None:
#             self.compute_rel_idf()
        if 'ARG' in rel: return 1
        return 2
    
    def concept_difficulty(self, concept):
        if self.concept_idf is None:
            self.compute_concept_idf()
        if re.match(r'^.*-\d+$',concept):
            return 1 + self.concept_idf.get(concept, 1)
        return 1

class DAMRV1(DAMR):

    def _get_nodes_depth(self, amr):
        depths = {amr.root:1}
        while True:
            added=False
            for node, depth in list(depths.items()):
                for _,tgt in amr.edges.get(node,[]):
                    if tgt not in depths:
                        depths[tgt] = depth + 1
                        added = True
            if not added: break
        # this should not happen   
        for node in amr.nodes:
            if node not in depths:
                depths[node]=1
        return depths
    
    def get_amr_difficulty(self, amr):
        difficulty = 0
        nodes_depth = self._get_nodes_depth(amr)

        for node in amr.nodes:
            concept = amr.name2concept[node]
            difficulty += self.concept_difficulty(concept) * nodes_depth[node] ** 2

        return difficulty


class DAMRV2(DAMR):

    def get_amr_difficulty(self, amr):
        difficulty = 0
        for src,tgts in amr.edges.items():
            src_c = amr.name2concept[src]
            for rel,tgt in tgts:
                tgt_c = amr.name2concept[tgt]
                difficulty+= self.rel_difficulty(rel) * (self.concept_difficulty(src_c) + self.concept_difficulty(tgt_c))
        
        return difficulty

DATA_LOADERS = {
    "DataLoader":DataLoader,
    "DefaultDataLoader":DataLoader
}

def register_dataloader(name=None):
    def register(dataloader_class):
        cname = name or dataloader_class.__name__
        if cname in DATA_LOADERS: raise ValueError('duplicate dataloader name %s'%cname)
        DATA_LOADERS[cname] = dataloader_class
        return dataloader_class
    return register

@register_dataloader()
class DAMRV1_CompetenceBasedCurriculumDataLoader(
        DAMRV1,
        CompetenceBasedCurriculumDataIter,
        Curriculum,DataLoader):
    pass

@register_dataloader()
class DAMRV2_CompetenceBasedCurriculumDataLoader(
        DAMRV2,
        CompetenceBasedCurriculumDataIter,
        Curriculum,DataLoader):
    pass

@register_dataloader()
class DAMRV2_BatchCompetenceBasedCurriculumDataLoader(
        DAMRV2,
        BatchCompetenceBasedCurriculumDataIter,
        Curriculum,DataLoader):
    pass

@register_dataloader()
class DAMRV2_CompetenceBasedMeanCurriculumDataLoader(
        DAMRV2,
        CompetenceBasedCurriculumDataIter,
        MeanCurriculum,DataLoader):
    pass

@register_dataloader()
class DAMRR0V2_CompetenceBasedCurriculumDataLoader(
        DAMRR0,
        DAMRV2,
        CompetenceBasedCurriculumDataIter,
        Curriculum,DataLoader):
    pass

@register_dataloader()
class DAMRV1_OrderedCurriculumDataLoader(
        DAMRV1,
        OrderedCurriculumDataIter, 
        Curriculum,DataLoader):
    pass

@register_dataloader()
class DAMRV2_OrderedCurriculumDataLoader(
        DAMRV2,
        OrderedCurriculumDataIter, 
        Curriculum,DataLoader):
    pass

@register_dataloader()
class DAMRV2_WaveCurriculumDataLoader(
        DAMRV2,
        WaveCurriculumDataIter, 
        Curriculum,DataLoader):
    pass
