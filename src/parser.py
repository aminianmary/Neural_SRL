import os
import pickle
import time
import utils
from optparse import OptionParser

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="conll_train", help="Annotated CONLL train file", metavar="FILE", default=None)
    parser.add_option("--dev", dest="conll_dev", help="Annotated CONLL dev file", metavar="FILE", default='')
    parser.add_option("--input", dest="input", help="Annotated CONLL test file", metavar="FILE", default=None)
    parser.add_option("--inputdir", dest="inputdir", help="Directory containing test files", metavar="FILE",
                      default=None)
    parser.add_option("--outputdir", dest="outputdir", help="Directory containing output files", metavar="FILE",
                      default=None)
    parser.add_option("--output", dest="output", help="output file", metavar="FILE", default=None)
    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params")
    parser.add_option("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE", default="model")
    parser.add_option("--d_w", type="int", dest="d_w", default=100)
    parser.add_option("--d_h", type="int", dest="d_h", default=512)
    parser.add_option("--d_c", type="int", dest="d_c", help="character embedding dimension", default=50)
    parser.add_option("--d_cw", type="int", dest="d_cw", help="character lstm dimension for lemma", default=100)
    parser.add_option("--d_pw", type="int", dest="d_pw", help="character lstm dimension for pos", default=100)
    parser.add_option("--d_r", type="int", dest="d_r", default=128)
    parser.add_option("--k", type="int", dest="k", default=4)
    parser.add_option("--lem_char_k", type="int", dest="lem_char_k", default=1)
    parser.add_option("--pos_char_k", type="int", dest="pos_char_k", default=1)
    parser.add_option("--batch", type="int", dest="batch", default=10000)
    parser.add_option("--alpha", type="float", dest="alpha", default=0.25)
    parser.add_option("--beta2", type="float", dest="beta2", default=0.999)
    parser.add_option("--beta1", type="float", dest="beta1", default=0.9)
    parser.add_option("--eps", type="float", dest="eps", default=0.00000001)
    parser.add_option("--sen_cut", type="int", dest="sen_cut", default=100)
    parser.add_option("--learning_rate", type="float", dest="learning_rate", default=0.001)
    parser.add_option("--epochs", type="int", dest="epochs", default=30)
    parser.add_option("--outdir", type="string", dest="outdir", default="results")
    parser.add_option("--dynet-autobatch", type="int", default=1)
    parser.add_option("--dynet-mem", type="int", default=10240)
    parser.add_option("--save_epoch", action="store_true", dest="save_epoch", default=False, help='Save each epoch.')
    parser.add_option("--region", action="store_false", dest="region", default=True, help='Use predicate boolean flag.')
    parser.add_option("--dynet-gpu", action="store_true", dest="--dynet-gpu", default=False,
                      help='Use GPU instead of cpu.')

    (options, args) = parser.parse_args()
    print 'Using external embedding:', options.external_embedding
    from srl import SRLLSTM

    if options.conll_train:
        print 'Preparing vocab'
        print options
        train_data = list(utils.read_conll(options.conll_train))
        words, pWords, pos, roles, chars = utils.vocab(train_data)
        with open(os.path.join(options.outdir, options.params), 'w') as paramsfp:
            pickle.dump((words, pWords, pos, roles, chars, options), paramsfp)
        print 'Finished collecting vocab'

        print 'Initializing blstm srl:'
        parser = SRLLSTM(words, pWords, pos, roles, chars, options)
        best_f_score = 0.0

        max_len = max([len(d) for d in train_data])
        min_len = min([len(d) for d in train_data])
        buckets = [list() for i in range(min_len, max_len)]
        for d in train_data:
            buckets[len(d) - min_len - 1].append(d)
        buckets = [x for x in buckets if x != []]

        for epoch in xrange(options.epochs):
            print 'Starting epoch', epoch
            print 'best F-score before starting the epoch: ' + str(best_f_score)
            best_f_score = parser.Train(utils.get_batches(buckets, parser, True, options.sen_cut), epoch, best_f_score, options)
            print 'best F-score after finishing the epoch: ' + str(best_f_score)

    sen_cut = options.sen_cut

    if options.input and options.output:
        with open(os.path.join(options.outdir, options.params), 'r') as paramsfp:
            words, lemmas, pos, roles, chars, stored_opt = pickle.load(paramsfp)
        stored_opt.external_embedding = options.external_embedding
        parser = SRLLSTM(words, lemmas, pos, roles, chars, stored_opt)
        parser.Load(os.path.join(options.outdir, options.model))
        ts = time.time()
        pred = list(parser.Predict(options.input, sen_cut))
        te = time.time()
        utils.write_conll(options.output, pred)
        print 'Finished predicting test', te - ts

    if options.inputdir and options.outputdir:
        with open(os.path.join(options.outdir, options.params), 'r') as paramsfp:
            words, lemmas, pos, roles, chars, stored_opt = pickle.load(paramsfp)
        stored_opt.external_embedding = options.external_embedding
        parser = SRLLSTM(words, lemmas, pos, roles, chars, stored_opt)
        parser.Load(os.path.join(options.outdir, options.model))
        ts = time.time()
        for dir, subdir, files in os.walk(options.inputdir):
            for f in files:
                print 'predicting ' + os.path.join(dir, f)
                pred = list(parser.Predict(os.path.join(dir, f)), sen_cut)

                utils.write_conll(options.outputdir + '/' + f + '.srl', pred)
        te = time.time()
        print 'Finished predicting test', te - ts