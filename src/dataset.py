import sys, os, time
import argparse
import re, csv
import pickle
import numpy as np
from transformers import BertTokenizer
import spacy
from spacy.tokens import Token
from spacy.tokenizer import Tokenizer
import task_utils
from tqdm import tqdm
import copy

global spacy_parser, tokenizer
Token.set_extension('tid', default=0)
spacy_parser = spacy.load("en_core_web_sm", disable=['ner'])
spacy_parser.tokenizer = Tokenizer(spacy_parser.vocab)
tokenizer = {
    'en-cased': BertTokenizer.from_pretrained('bert-base-cased'),
    'en-uncased': BertTokenizer.from_pretrained('bert-base-uncased'),
}

class InputRawExample(object):
    def __init__(self, text, label):
        self.text = text
        self.label = label

class Graph(object):
    """docstring for Graph"""
    def __init__(self, n):
        super(Graph, self).__init__()
        self.n = n
        self.link_list = []
        self.vis = [0] * self.n
        for i in range(self.n):
            self.link_list.append([])

    def add_edge(self, u, v):
        if u == v:
            return
        self.link_list[u].append(v)
        self.link_list[v].append(u)

    def bfs(self, start, dist):
        que = [start]
        self.vis[start] = 1
        for _ in range(dist):
            que2 = []
            for u in que:
                #self.vis[u] = 1
                for v in self.link_list[u]:
                    if self.vis[v]:
                        continue
                    que2.append(v)
                    self.vis[v] = 1
            que = copy.deepcopy(que2)
            

    def solve(self, start, dist):
        self.vis = [0] * self.n
        self.bfs(start, dist)
        self.vis[0] = 1
        return copy.deepcopy(self.vis)

def process(args, text, label):
    # text: str
    # label: list[str] or int
    global spacy_parser, tokenizer
    if args.lang == 'en':
        local_tokenizer = tokenizer['en-uncased'] if args.do_lower_case else tokenizer['en-cased']

        while '  ' in text:
            text = text.replace('  ', ' ')
        doc = spacy_parser(text)

        tokens = ['[CLS]']
        for token in doc:
            token._.tid = len(tokens)
            tokens.append(token.text)
        
        G = Graph(len(tokens))
        for token in doc:
            if token.dep_ == 'ROOT':
                continue
            G.add_edge(token._.tid, token.head._.tid)


        ntokens = []
        ws = []

        for i, token in enumerate(tokens):
            if token == '[CLS]':
                ntokens.append(token)
                ws.append(1)
            else:
                sub_tokens = local_tokenizer.tokenize(token)
                ws.append(len(sub_tokens))
                for j, st in enumerate(sub_tokens):
                    ntokens.append(st)

        dep_att_mask = []
        for i, token in enumerate(tokens):
            vis = G.solve(i, args.dist)
            
            if i-1>=0:
                vis_tmp = G.solve(i-1, args.dist)
                for j in range(len(vis_tmp)):
                    vis[j] |= vis_tmp[j]

            if i+1<len(tokens):
                vis_tmp = G.solve(i+1, args.dist)
                for j in range(len(vis_tmp)):
                    vis[j] |= vis_tmp[j]
            
            mask = []
            for j in range(len(vis)):
                for k in range(ws[j]):
                    mask.append(vis[j])

            assert len(ntokens) == len(mask), ntokens
            mask.append(1)

            for k in range(ws[i]):
                dep_att_mask.append(mask)

        ntokens.append('[SEP]')
        if isinstance(label, list):
            labels = []
            loss_mask = []
            for i, token in enumerate(tokens):
                if token == '[CLS]':
                    labels.append('O')
                    loss_mask.append(0)
                else:
                    for j in range(ws[i]):
                        labels.append(label[i-1])
                        loss_mask.append(1 if j==0 else 0)
            labels.append('O')
            loss_mask.append(0)
            label_s2i = {}
            for i, s in enumerate(task_utils.task_processors[args.task]().get_labels()):
                label_s2i[s] = i
            labels = [label_s2i[s] for s in labels]
        else:
            labels = [label]

        dep_att_mask.append([1] * len(ntokens))
        for j in range(len(dep_att_mask[0])):
            dep_att_mask[0][j] = 1

        quote_idx = []
        for i, w in enumerate(ntokens):
            w = w.lower()
            if w in '!\"#$%&\'()*+,.-/:;<=>?@[\\]^_':
                quote_idx.append(i)

        for i in range(len(ntokens)):
            for j in quote_idx:
                dep_att_mask[i][j] = 1
        
        input_ids = local_tokenizer.convert_tokens_to_ids(ntokens)
    else:
        raise KeyError(args.lang)

    example = {
        'input_ids': input_ids,
        'dep_att_mask': dep_att_mask,
        'labels': labels,
    }
    if isinstance(label, list):
        example['loss_mask'] = loss_mask
    return example

def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            lines.append(line)
        return lines

def _read_conll_format_file(input_file):
    lines = []
    with open(input_file, 'r', encoding='utf-8') as f:
        words = []
        labels = []
        for line in f:
            contends = line.strip()
            feature_vector = contends.split(' ')
            word = feature_vector[0].strip()
            label = feature_vector[-1].strip()
            if len(contends) == 0:
                w = ' '.join(words)
                l = ' '.join(labels)
                lines.append((w, l))
                words = []
                labels = []
                continue                    
            #word = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a'!'#$%&\'()*+,-./:;<=>?@????????????????????????????????????????'???[\\]^_`{|}~])","",word)
            if len(word) == 0 or len(label) == 0:
                continue
            words.append(word)
            labels.append(label)
    return lines

def main(args):
    raw_examples = []
    if args.task == 'cola':
        lines = _read_tsv(args.data_file)
        for line in lines:
            text = line[3].lower() if args.do_lower_case else line[3]
            raw_examples.append(InputRawExample(text, int(line[1])))
    elif args.task == 'sst-2':
        lines = _read_tsv(args.data_file)
        for line in lines[1:]:
            text = line[0].lower() if args.do_lower_case else line[0]
            raw_examples.append(InputRawExample(text, int(line[1])))
    elif args.task == 'fce':
        lines = _read_conll_format_file(args.data_file)
        for w, l in lines:
            text = w.lower() if args.do_lower_case else w
            raw_examples.append(InputRawExample(text, l.split(' ')))
    else:
        raise KeyError(args.task)
    examples = []
    for item in tqdm(raw_examples, desc='Convert'):
        examples.append(process(args, item.text, item.label))

    filename = args.data_file.split('/')[-1]
    pickle.dump(examples, open('./%s.%s.d%d.pkl'%(args.task, filename, args.dist), 'wb'))


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="cola", type=str)
    p.add_argument("--data_file", default="cola.tsv", type=str)
    p.add_argument("--lang", default="en", type=str)
    p.add_argument("--do_lower_case", action="store_true")
    p.add_argument("--dist", default=3, type=int)
    args = p.parse_args()
    main(args)
