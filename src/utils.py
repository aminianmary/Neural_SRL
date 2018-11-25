from collections import Counter, defaultdict
import re, codecs, random
import numpy as np

class ConllStruct:
    def __init__(self, entries, predicates):
        self.entries = entries
        self.predicates = predicates

    def __len__(self):
        return len(self.entries)

class ConllEntry:
    def __init__(self, id, form, lemma, pos, sense='_', parent_id=-1, relation='_', predicateList=dict(),
                 is_pred=False):
        self.id = id
        self.form = form
        self.lemma = lemma[0:50]
        self.norm = normalize(form)
        self.lemmaNorm = normalize(lemma)[0:50]
        self.pos = pos.upper()
        self.parent_id = parent_id
        self.relation = relation
        self.predicateList = predicateList
        self.sense = sense
        self.is_pred = is_pred

    def __str__(self):
        entry_list = [str(self.id+1), self.form, self.lemma, self.lemma, self.pos, self.pos, '_', '_',
                      self.parent_id,
                      self.parent_id, self.relation, self.relation,
                      'Y' if self.is_pred == True else '_',
                    self.sense if self.sense != '?' else '_']
        for p in self.predicateList.values():
            entry_list.append(p)
        return '\t'.join(entry_list)

def vocab(sentences, min_count=2):
    wordsCount = Counter()
    posCount = Counter()
    semRelCount = Counter()
    lemma_count = Counter()
    chars = set()

    for sentence in sentences:
        wordsCount.update([node.norm for node in sentence.entries])
        posCount.update([node.pos for node in sentence.entries])
        for node in sentence.entries:
            if node.predicateList == None:
                continue
            if node.is_pred:
                lemma_count.update([node.lemma])
            for pred in node.predicateList.values():
                if pred!='?':
                    semRelCount.update([pred])
            for c in list(node.norm):
                    chars.add(c.lower())

    words = set()
    for w in wordsCount.keys():
        if wordsCount[w] >= min_count:
            words.add(w)
    lemmas = set()
    for l in lemma_count.keys():
        if lemma_count[l] >= min_count:
            lemmas.add(l)
    return (list(words), list(lemmas),
            list(posCount), list(semRelCount.keys()), list(chars))

def read_conll(fh):
    sentences = codecs.open(fh, 'r').read().strip().split('\n\n')
    read = 0
    for i,sentence in enumerate(sentences):
        print i
        words = []
        predicates = list()
        entries = sentence.strip().split('\n')
        for entry in entries:
            spl = entry.split('\t')
            # print len(spl)
            predicateList = dict()
            is_pred = False
            if spl[12] == 'Y':
                is_pred = True
                predicates.append(int(spl[0]) - 1)

            for i in range(14, len(spl)):
                predicateList[i - 14] = spl[i]

            words.append(
                ConllEntry(int(spl[0]) - 1, spl[1], spl[3], spl[5], spl[13], spl[9], spl[11], predicateList,
                           is_pred))
        read += 1
        yield ConllStruct(words, predicates)
    print read, 'sentences read.'

def write_conll(fn, conll_structs):
    with codecs.open(fn, 'w') as fh:
        for conll_struct in conll_structs:
            for i in xrange(len(conll_struct.entries)):
                entry = conll_struct.entries[i]
                fh.write(str(entry))
                fh.write('\n')
            fh.write('\n')

numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")
urlRegex = re.compile("((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)")

def normalize(word):
    return '<NUM>' if numberRegex.match(word) else ('<URL>' if urlRegex.match(word) else word.lower())

def get_batches(buckets, model, is_train, sen_cut):
    d_copy = [buckets[i][:] for i in range(len(buckets))]
    if is_train:
        for dc in d_copy:
            random.shuffle(dc)
    mini_batches = []
    batch, pred_ids, cur_len, cur_c_len, cur_pred_c_len = [], [], 0, 0, 0
    b = model.options.batch if is_train else model.options.dev_batch_size
    for dc in d_copy:
        for d in dc:
            if (is_train and len(d)<=sen_cut) or not is_train:
                for p, predicate in enumerate(d.predicates):
                    batch.append(d.entries)
                    pred_ids.append([p,predicate])
                    cur_c_len = max(cur_c_len, max([len(w.norm) for w in d.entries]))
                    cur_len = max(cur_len, len(d))
                    cur_pred_c_len = max(cur_pred_c_len, len(d.entries[predicate].norm))

            if cur_len * len(batch) >= b:
                add_to_minibatch(batch, pred_ids, cur_c_len, cur_len, cur_pred_c_len, mini_batches, model)
                batch, pred_ids, cur_len, cur_c_len, cur_pred_c_len = [], [], 0, 0, 0

    if len(batch)>0 and not is_train:
        add_to_minibatch(batch, pred_ids, cur_c_len, cur_len, cur_pred_c_len, mini_batches, model)
    if is_train:
        random.shuffle(mini_batches)
    return mini_batches


def add_to_minibatch(batch, pred_ids, cur_c_len, cur_len, cur_pred_c_len, mini_batches, model):
    #formed to work in the batch mode
    words = np.array([np.array(
        [model.words.get(batch[i][j].norm, 0) if j < len(batch[i]) else model.PAD_index for i in
         range(len(batch))]) for j in range(cur_len)])
    pwords = np.array([np.array(
        [model.x_pe_dict.get(batch[i][j].norm, 0) if j < len(batch[i]) else model.PAD_index for i in
         range(len(batch))]) for j in range(cur_len)])
    lemmas = np.array([np.array(
        [(model.pred_lemmas.get(batch[i][j].lemma, 0) if pred_ids[i][1]==j else model.NO_LEMMA_index)if j < len(batch[i])
         else model.PAD_index for i in range(len(batch))]) for j in range(cur_len)]) if model.use_lemma else None
    pos = np.array([np.array(
        [model.pos.get(batch[i][j].pos, 0) if j < len(batch[i]) else model.PAD_index for i in
         range(len(batch))]) for j in range(cur_len)]) if model.use_pos else None
    roles = np.array([np.array(
        [model.roles.get(batch[i][j].predicateList[pred_ids[i][0]], 0) if j < len(batch[i]) else model.PAD_index for i in
         range(len(batch))]) for j in range(cur_len)])
    chars = [list() for _ in range(cur_c_len)]
    for c_pos in range(cur_c_len):
        ch = [model.PAD_index] * (len(batch) * cur_len)
        offset = 0
        for w_pos in range(cur_len):
            for sen_position in range(len(batch)):
                if w_pos < len(batch[sen_position]) and c_pos < len(batch[sen_position][w_pos].norm):
                    ch[offset] = model.chars.get(batch[sen_position][w_pos].norm[c_pos].lower(), 0)
                offset += 1
        chars[c_pos] = np.array(ch)
    chars = np.array(chars)

    pred_chars = [list() for _ in range(cur_pred_c_len)]
    for c_pos in range(cur_pred_c_len):
        ch = [model.PAD_index] * len(batch)
        for sen_position in range(len(batch)):
            if c_pos < len(batch[sen_position][pred_ids[sen_position][1]].norm):
                ch[sen_position] = model.chars.get(batch[sen_position][pred_ids[sen_position][1]].norm[c_pos].lower(), 0)
        pred_chars[c_pos] = np.array(ch)
    pred_chars = np.array(pred_chars)

    pred_flags = np.array([np.array([(1 if pred_ids[i][1] == j else 0)
                                     if j < len(batch[i]) else 0 for i in range(len(batch))]) for j in range(cur_len)])
    pred_lemmas = np.array([model.pred_lemmas.get(batch[i][pred_ids[i][1]].lemma, 0) for i in range(len(batch))])
    pred_index = np.array([pred_ids[i][1] for i in range(len(batch))])
    masks = np.array([np.array([1 if j < len(batch[i]) and batch[i][j].predicateList[pred_ids[i][0]]!='?' else 0 for i in range(len(batch))]) for j in range(cur_len)])
    mini_batches.append((words, pwords, lemmas, pos, roles, chars, pred_chars, pred_flags, pred_lemmas, pred_index, masks))

def get_scores(fp):
    labeled_f = 0
    unlabeled_f = 0
    line_counter =0
    with codecs.open(fp, 'r') as fr:
        for line in fr:
            line_counter+=1
            if line_counter == 10:
                spl = line.strip().split(' ')
                labeled_f= spl[len(spl)-1]
            if line_counter==13:
                spl = line.strip().split(' ')
                unlabeled_f = spl[len(spl) - 1]
    return (labeled_f, unlabeled_f)

def replace_unk_with_system_output (output, gold, replaced_file):
    output_lines = codecs.open(output,'r').read().strip().split('\n')
    gold_lines = codecs.open(gold,'r').read().strip().split('\n')

    if len(output_lines)!= len(gold_lines):
        print 'number of lines does not match between the system output and gold file!'

    with codecs.open(replaced_file,'w') as writer:
        for i in xrange(len(output_lines)):
            if output_lines[i]!= '':
                o_fields = output_lines[i].split()[12:]
                g_fileds = gold_lines[i].split()[12:]

                for j in xrange(len(o_fields)):
                    if o_fields[j] == '?':
                        o_fields[j]= g_fileds[j]
                writer.write('\t'.join(output_lines[i].split()[:12])+ '\t' + '\t'.join(o_fields)+ '\n')
            else:
                writer.write('\n')
        writer.flush()
        writer.close()
    return replaced_file

def replace_unk_with_null (output):
    output_lines = codecs.open(output,'r').read().strip().split('\n')
    replaced_file = output +'.unk_replaced_with_null'

    with codecs.open(replaced_file,'w') as writer:
        for i in xrange(len(output_lines)):
            l = output_lines[i]
            if l != '':
                o_fields = l .split()[12:]
                for j in xrange(len(o_fields)):
                    if o_fields[j] == '?':
                        o_fields[j]= '_'
                writer.write('\t'.join(l.split()[:12])+ '\t' + '\t'.join(o_fields)+ '\n')
            else:
                writer.write('\n')
        writer.flush()
        writer.close()
    return replaced_file
