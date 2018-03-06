from dynet import *
from utils import read_conll, get_batches, get_scores, write_conll, eval_sense, get_predicates_list
import time, random, os,math
import numpy as np


class SRLLSTM:
    def __init__(self, words, pWords, plemmas, pos, senses, chars, sense_mask, options):
        self.model = Model()
        self.options = options
        self.batch_size = options.batch
        self.trainer = AdamTrainer(self.model, options.learning_rate, options.beta1, options.beta2, options.eps)
        self.trainer.set_clip_threshold(1.0)
        self.unk_id = 0
        self.PAD = 1
        self.NO_LEMMA = 2
        self.words = {word: ind + 2 for ind,word in enumerate(words)}
        self.pWords = {word: ind + 2 for ind,word in enumerate(pWords)}
        self.plemmas = {word: ind + 2 for ind,word in enumerate(plemmas)}
        self.pos = {p: ind + 2 for ind, p in enumerate(pos)}
        self.ipos = ['<UNK>', '<PAD>'] + pos
        senses = ['<UNK>'] + senses
        self.senses = {s: ind for ind, s in enumerate(senses)}
        self.isenses = senses
        self.sense_mask = sense_mask
        self.char_dict = {c: i + 2 for i, c in enumerate(chars)}
        self.d_w = options.d_w
        self.d_cw = options.d_cw
        self.d_pw = options.d_pw
        self.d_h = options.d_h
        self.d_r = options.d_r
        self.k = options.k
        self.lem_char_k = options.lem_char_k
        self.pos_char_k = options.pos_char_k
        self.alpha = options.alpha
        self.external_embedding = None
        self.x_pe = None
        self.region = options.region
        external_embedding_fp = open(options.external_embedding, 'r')
        external_embedding_fp.readline()
        self.external_embedding = {line.split(' ')[0]: [float(f) for f in line.strip().split(' ')[1:]] for line in external_embedding_fp}
        external_embedding_fp.close()
        self.edim = len(self.external_embedding.values()[0])
        self.noextrn = [0.0 for _ in xrange(self.edim)]
        self.x_pe_dict = {word: i + 2 for i, word in enumerate(self.external_embedding)}
        self.x_pe = self.model.add_lookup_parameters((len(self.external_embedding) + 2, self.edim))
        for word, i in self.x_pe_dict.iteritems():
            self.x_pe.init_row(i, self.external_embedding[word])
        self.x_pe.init_row(0,self.noextrn)
        self.x_pe.init_row(1,self.noextrn)
        self.x_pe.set_updated(False)
        print 'Load external embedding. Vector dimensions', self.edim

        self.inp_dim = self.d_w + self.d_cw + self.d_pw + (self.edim if self.external_embedding is not None else 0)
        self.lemma_char_lstm = BiRNNBuilder(self.lem_char_k, options.d_c, options.d_cw, self.model, VanillaLSTMBuilder)
        self.pos_char_lstm = BiRNNBuilder(self.pos_char_k, options.d_c, options.d_pw, self.model, VanillaLSTMBuilder)
        self.deep_lstms = BiRNNBuilder(self.k, self.inp_dim, 2*self.d_h, self.model, VanillaLSTMBuilder)
        self.x_re = self.model.add_lookup_parameters((len(self.words) + 2, self.d_w))
        self.ce = self.model.add_lookup_parameters((len(chars) + 2, options.d_c)) # lemma character embedding
        self.W = self.model.add_parameters((len(self.isenses), self.d_h * 2))
        self.b = self.model.add_parameters((len(self.isenses)), init = ConstInitializer(0))

    def Save(self, filename):
        self.model.save(filename)

    def Load(self, filename):
        self.model.populate(filename)

    def rnn(self, words, pwords, chars):
        cembed = [lookup_batch(self.ce, c) for c in chars]
        lem_char_fwd, lem_char_bckd = self.lemma_char_lstm.builder_layers[0][0].initial_state().transduce(cembed)[-1], \
                              self.lemma_char_lstm.builder_layers[0][1].initial_state().transduce(reversed(cembed))[-1]
        pos_char_fwd, pos_char_bckd = self.pos_char_lstm.builder_layers[0][0].initial_state().transduce(cembed)[-1], \
                              self.pos_char_lstm.builder_layers[0][1].initial_state().transduce(reversed(cembed))[-1]
        lem_crnn = reshape(concatenate_cols([lem_char_fwd, lem_char_bckd]), (self.d_cw, words.shape[0] * words.shape[1]))
        pos_crnn = reshape(concatenate_cols([pos_char_fwd, pos_char_bckd]), (self.d_pw, words.shape[0] * words.shape[1]))
        lem_cnn_reps = [list() for _ in range(len(words))]
        pos_cnn_reps = [list() for _ in range(len(words))]

        # first dim: word position; second dim: sentence number.
        for i in range(words.shape[0]):
            lem_cnn_reps[i] = pick_batch(lem_crnn, [i * words.shape[1] + j for j in range(words.shape[1])], 1)

        for i in range(words.shape[0]):
            pos_cnn_reps[i] = pick_batch(pos_crnn, [i * words.shape[1] + j for j in range(words.shape[1])], 1)

        inputs = [concatenate([lookup_batch(self.x_re, words[i]), lookup_batch(self.x_pe, pwords[i]),
                                pos_cnn_reps[i],
                               lem_cnn_reps[i]]) for i in range(len(words))]

        for fb, bb in self.deep_lstms.builder_layers:
            f, b = fb.initial_state(), bb.initial_state()
            fs, bs = f.transduce(inputs), b.transduce(reversed(inputs))
            inputs = [concatenate([f, b]) for f, b in zip(fs, reversed(bs))]
        return inputs

    def buildGraph(self, minibatch, is_train):
        words, pos, pwords, pos, pred_lemmas_index, chars, senses, masks = minibatch
        bilstms = self.rnn(words, pwords, chars)
        bilstms = [transpose(reshape(b, (b.dim()[0][0], b.dim()[1]))) for b in bilstms]
        senses, masks = senses.T, masks.T

        v_p = [bilstms[pred_lemmas_index[sen]][sen] for sen in range(senses.shape[0])]
        bv_p = concatenate_to_batch(v_p)
        scores = affine_transform([self.b.expr(), self.W.expr(), bv_p])
        if is_train:
            gold_senses = [senses[sen][pred_lemmas_index[sen]] for sen in range(senses.shape[0])]
            err = pickneglogsoftmax_batch(scores, gold_senses)
            output = sum_batches(err) / err.dim()[1]
        else:
            output = scores

        return output

    def decode(self, minibatches):
        outputs = [list() for _ in range(len(minibatches))]
        for b, batch in enumerate(minibatches):
            print 'batch '+ str(b)
            outputs[b] = self.buildGraph(batch, False).npvalue()
            if len(outputs[b].shape) == 1:
                outputs[b] = np.reshape(outputs[b], (outputs[b].shape[0], 1))
            renew_cg()
        print 'decoded all the batches! YAY!'
        outputs = np.concatenate(outputs, axis=1)
        return outputs.T

    def Train(self, mini_batches, epoch, best_f_score, options):
        print 'Start time', time.ctime()
        start = time.time()
        iters = 0
        dev_path = options.conll_dev

        part_size = max(len(mini_batches)/5, 1)
        part = 0
        best_part = 0

        for b, mini_batch in enumerate(mini_batches):
            sum_errs = self.buildGraph(mini_batch, True)
            loss = sum_errs.scalar_value()
            sum_errs.backward()
            self.trainer.update()
            renew_cg()
            print 'loss:', loss/(b+1), 'time:', time.time() - start, 'progress',round(100*float(b+1)/len(mini_batches),2),'%'
            start = time.time()
            iters+=1

            if (b+1)%part_size==0:
                part+=1

                if dev_path != '':
                    start = time.time()
                    write_conll(os.path.join(options.outdir, options.model) + str(epoch + 1) + "_" + str(part)+ '.txt',
                                      self.Predict(dev_path, options.sen_cut, options.use_lemma, options.use_default_sense))
                    accuracy = eval_sense(dev_path, os.path.join(options.outdir, options.model) + str(epoch + 1) + "_" + str(part)+ '.txt')

                    if float(accuracy) > best_f_score:
                        self.Save(os.path.join(options.outdir, options.model))
                        best_f_score = accuracy
                        best_part = part

        print 'best part on this epoch: '+ str(best_part)
        return best_f_score

    def Predict(self, conll_path, sen_cut, use_lemma, use_default_sense):
        print 'starting to decode...'
        dev_buckets = [list()]
        dev_data = list(read_conll(conll_path))
        for d in dev_data:
            dev_buckets[0].append(d)
        minibatches = get_batches(dev_buckets, self, False, sen_cut)
        outputs = self.decode(minibatches)
        pwords = self.pWords if not use_lemma else self.plemmas
        dev_predicate_words = get_predicates_list(dev_data, pwords, use_lemma, use_default_sense)
        outputs_ = [outputs[i] + self.sense_mask[dev_predicate_words[i]] for i in range(len(outputs))]
        results = [self.isenses[np.argmax(outputs_[i])] for i in range(len(outputs))]
        offset = 0
        for iSentence, sentence in enumerate(dev_data):
            for p in sentence.predicates:
                res = results[offset] if results[offset] != '<UNK>' else sentence.entries[p].norm+'.01'
                sentence.entries[p].sense = res
                offset+=1
            yield sentence