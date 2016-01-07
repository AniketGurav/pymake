import sys, os
from itertools import chain
from string import Template

from local_utils import *
from vocabulary import Vocabulary, parse_corpus
from util.frontend import frontEndBase

sys.path.insert(1, '../../gensim')
import gensim
from gensim.models import ldamodel, ldafullbaye
Models = { 'ldamodel': ldamodel, 'ldafullbaye': ldafullbaye, 'hdp': 1}

############################################################
############################################################
#### Aim at be a frontend for corpus data manipulation.
####   * First, purpose is to be the frontend for model/algorithm input,
####   * Second, frontend for data observation. States of corpus or results analysis,
####   * Third, operate on corpus various operation as filtering, merging etc.
####
####  load_corpus->load_text_corpus->text_loader
####  (frontent) -> (choice) -> (adapt preprocessing)

# review that:
#    * separate better load / save and preprocessing (input can be file or array...)
#    * view of file confif.... and path creation....

class frontEndText(frontEndBase):

    def __init__(self, config):
        config['bdir'] = os.path.join(config['bdir'], 'text')
        super(frontEndText, self).__init__(config)

    def load_corpus(self, corpus_name=None):
        self.load_text_corpus(corpus_name)
        # @DEBUG
        if self.N > self.corpus.shape[0]:
            self.N = self.config['N'] = None
        return self.corpus

    ### Get and prepropress text
    #   See Vocabulary class...
    #   * Tokenisation from scratch
    #   * Stop Word from scratch
    #   * Lemmatization from Wornet
    #   * Load or Save in a Gensim context
    #       - Load has priority over Save
    # @Debug: There is a convertion to gensim corpus to use the serialization library and then back to scipy corpus.
    #   Can be avoided by using our own library of serialization, using Gensim if needed ?!
    def textloader(self, target, bdir=None, corpus_name="", n=None):
        if type(target) is str and os.path.isfile(target):
            bdir = os.path.dirname(target)
        elif bdir is None:
            bdir = self.basedir
        fn = 'corpus'
        if n:
            fn += str(n)
        elif type(target) is not str:
            n = len(target)
            fn += str(n)

        if corpus_name:
            fname = bdir + '/'+fn+'_' + corpus_name + '.mm'
        else:
            fname = bdir + '/'+fn+'.mm'

        if self._load_corpus and os.path.isfile(fname):
            corpus = gensim.corpora.MmCorpus(fname)
            corpus = gensim.matutils.corpus2csc(corpus, dtype=int).T
            id2word = dict(gensim.corpora.dictionary.Dictionary.load_from_text(fname + '.dico'))
        else:
            print 're-Building Corpus...'
            raw_data, id2word = parse_corpus(target)

            # Corpus will be in bag of words format !
            if type(raw_data) is list:
                voca = Vocabulary(exclude_stopwords=True)
                corpus = [voca.doc_to_bow(doc) for doc in raw_data]
                corpus = gensim.matutils.corpus2csc(corpus, dtype=int).T # Would be faster with #doc #term #nnz
            else:
                corpus = raw_data

            if self._save_corpus:
                make_path(bdir)
                _corpus = gensim.matutils.Sparse2Corpus(corpus, documents_columns=False)
                voca_gensim = gensim.corpora.dictionary.Dictionary.from_corpus(_corpus, id2word)
                voca_gensim.save_as_text(fname+'.dico')
                gensim.corpora.MmCorpus.serialize(fname=fname, corpus=_corpus)
                #@Debug how to get the corpus from list of list ?
                #_corpus = gensim.corpora.MmCorpus(fname)

        return corpus, id2word

    def load_text_corpus(self, corpus_name=None):
        config = self.config
        corpus_name = corpus_name or self.corpus_name
        bdir = self.basedir
        self.corpus_name = corpus_name
        corpus_t = None
        if corpus_name == 'lucene':
            raise NotImplementedError
            #searcher = warm_se(config)
            #q = config.get('q'); q['limit'] = config['limit_train']
            #id2word = searcher.get_id2word()
            #corpus = searcher.self.parse_corpus(q, vsm=config['vsm'], chunk=1000, batch=True)
        elif corpus_name == '20ngroups_sklearn':
            from sklearn.datasets import fetch_20newsgroups
            ngroup_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=None)
            ngroup_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=None)
            train_data = ngroup_train.data
            test_data = ngroup_test.data
            #corpus, id2word = self.textloader(train_data, bdir=bdir, corpus_name='train', n=config.get('N'))
            corpus, id2word = self.textloader(train_data, bdir=bdir, corpus_name='train')
            corpus_t, id2word_t = self.textloader(test_data, bdir=bdir, corpus_name='test')

            K = self.K
            #################
            ### Group Control
            test_classes = ngroup_test.target
            train_classes = ngroup_train.target
            if K == 6 and len(ngroup_test.target_names) != 6:
                # Wrap to subgroups
                target_names = ['comp', 'misc', 'rec', 'sci', 'talk', 'soc']
                map_ = dict([(0,5), (1,0), (2,0), (3,0), (4,0), (5,0), (6,1), (7,2), (8,2), (9,2), (10,2), (11,3), (12,3), (13,3), (14,3), (15,5), (16,4), (17,4), (18,4), (19,5)])
                test_classes = set_v_to(test_classes, map_)
                train_classes = set_v_to(train_classes, map_)
            else:
                target_names = ngroup_test.target_names
            C = len(target_names)

        elif corpus_name == 'wikipedia':
            # ? file type
            # Create
            command = './gensim/gensim/scripts/make_wikicorpus_ml.py '
            command += '/work/adulac/data/wikipedia/enwiki-latest-pages-articles.xml.bz2 ../PyNPB/data/wikipedia/wiki_en'
            os.system(command)
            # Load
            error = 'Load Wikipedia corpus'
            raise NotImplementedError(error)
        elif corpus_name == 'odp':
            # SVMlight file type
            from sklearn.datasets import load_svmlight_files, load_svmlight_file
            fn_train = os.path.join(bdir, 'train.txt')
            fn_test = os.path.join(bdir, 'test.txt')
            # More feature in test than train !!!
            corpus, train_classes = load_svmlight_file(fn_train)
            corpus_t, test_classes = load_svmlight_file(fn_test)
            id2word = None
        elif corpus_name in ('reuter50', 'nips12', 'nips', 'enron', 'kos', 'nytimes', 'pubmed') or corpus_name == '20ngroups' :
            # DOC_ID FEAT_ID COUNT file type
            corpus, id2word = self.textloader(bdir, corpus_name=corpus_name)
        else:
            raise ValueError('Which corpus to Load ?')

        self.corpus = corpus
        self.id2word = id2word
        if corpus_t is None:
            pass
            #raise NotImplementedError('Corpus test ?')
        else:
            self.corpus_t = corpus_t

        return True

    def get_corpus_prop(self):
        prop =  super(frontEndText, self).get_corpus_prop()
        nnz = self.corpus.sum(),
        _nnz = self.corpus.sum(axis=1)
        dct = {'features': self.corpus.shape[1],
               'nnz': nnz,
               'nnz_mean': _nnz.mean(),
               'nnz_var': _nnz.var(),
               'train_size': None,
               'test_size': None,
              }
        prop.update(dct)
        return prop

    def template(self, dct):
        text_templ = '''###### $corpus_name
        Building: $time minutes
        Documents: $instances
        Nnz: $nnz
        Nnz mean: $nnz_mean
        Nnz var: $nnz_var
        Vocabulary: $features
        train: $train_size
        test: $test_size
        \n'''
        return super(frontEndText, self).template(dct, text_templ)

    def print_vocab(self, corpus, id2word):
        if id2word:
            return gensim.corpora.dictionary.Dictionary.from_corpus(corpus, id2word) #; print voca

    def shuffle_docs(self):
        self.shuffle_instances()

    # Debug
    def run_lda(self):
        pass
   #     # Cross Validation settings...
   #     #@DEBUG: do we need to remake the vocabulary ??? id2word would impact the topic word distribution ?
   #     if corpus_t is None:
   #         pass
   #         #take 80-20 %
   #         # remake vocab and shape !!!
   #         # manage downside
   #     try:
   #         total_corpus = len(corpus)
   #         total_corpus_t = len(corpus_t)
   #     except:
   #         total_corpus = corpus.shape[0]
   #         total_corpus_t = corpus.shape[0]
   #     if config.get('N'):
   #         N = config['N']
   #     else:
   #         N = total_corpus
   #     corpus = corpus[:N]
   #     n_percent = float(N) / total_corpus
   #     n_percent = int(n_percent * total_corpus_t) or 10
   #     heldout_corpus = corpus_t[:n_percent]

   #     ############
   #     ### Load LDA
   #     load = config['load_model']
   #     save = config['save_model']
   #     # Path for LDA model!
   #     bdir = '../PyNPB/data/'
   #     bdir = os.path.join(bdir,config.get('corpus'), config.get('bdir', ''))
   #     lda = lda_gensim(corpus, id2word=id2word, K=K, bdir=bdir, load=load, model=config['model'], alpha=config['hyper'], n=config['N'], heldout_corpus=heldout_corpus)
   #     lda.inference_time = datetime.now() - last_d
   #     last_d = ellapsed_time('LDA Inference -- '+config['model'], last_d)

   #     ##############
   #     ### Log Output
   #     print
   #     logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
   #     lda.print_topics(K)
   #     print

   #     ##############
   #     ### Prediction
   #     corpus_t = corpus
   #     if config['predict'] and true_classes is not None and C == K:
   #         true_classes = train_classes
   #         predict_class = []
   #         confusion_mat = np.zeros((K,C))
   #         startt = datetime.now()
   #         for i, d in enumerate(corpus_t):
   #             d_t = lda.get_document_topics(d, minimum_probability=0.01)
   #             t = max(d_t, key=lambda item:item[1])[0]
   #             predict_class.append(t)
   #             c = true_classes[i]
   #             confusion_mat[t, c] += 1
   #         last_d = ellapsed_time('LDA Prediction', startt)
   #         predict_class = np.array(predict_class)
   #         lda.confusion_matrix = confusion_mat

   #         map_kc = map_class2cluster_from_confusion(confusion_mat)
   #         #new_predict_class = set_v_to(predict_class, dict(map_kc))

   #         print "Confusion Matrix, KxC:"
   #         print confusion_mat
   #         print map_kc
   #         print [(k, target_names[c]) for k,c in map_kc]

   #         purity = confusion_mat.max(axis=1).sum() / len(corpus_t)
   #         print 'Purity (K=%s, C=%s, D=%s): %s' % (K, C, len(corpus_t), purity)

   #         #precision = np.sum(new_predict_class == true_classes) / float(len(predict_class)) # equal !!!
   #         precision = np.sum(confusion_mat[zip(*map_kc)]) / float(len(corpus_t))
   #         print 'Ratio Groups Control: %s' % (precision)

   #     if save:
   #         ## Too big
   #         lda.expElogbeta = None
   #         lda.sstats = None
   #         lda.save(lda.fname)

   #     if config.get('verbose'):
   #         #print lda.top_topics(corpus)
   #         for d in corpus:
   #             print lda.get_document_topics(d, minimum_probability=0.01)

   #     print
   #     print lda
   #     if type(corpus) is not list:
   #         print corpus
   #         print corpus_t
   #     self.print_vocab(corpus, id2word)

    ########################################################################"
    ### LDA Worker
    def lda_gensim(corpus=None, id2word=None, K=10, alpha='auto', save=False, bdir='tmp/', model='ldamodel', load=False, n=None, heldout_corpus=None, updatetype='batch'):
        try: n = len(corpus) if corpus is not None else n
        except: n = corpus.shape[0]
        fname = bdir + "/%s_%s_%s_%s.gensim" % ( model, str(K), alpha, n)
        if load:
            return Models[model].LdaModel.load(fname)

        if hasattr(corpus, 'tocsc'):
            # is csr sparse matrix
            corpus = corpus.tocsc()
            corpus = gsm.matutils.Sparse2Corpus(corpus, documents_columns=False)
            if heldout_corpus is not None:
                heldout_corpus = heldout_corpus.tocsc()
                heldout_corpus = gsm.matutils.Sparse2Corpus(heldout_corpus, documents_columns=False)
        elif isanparray:
            # up tocsc ??!!! no !
            dense2corpus
        # Passes is the iterations for batch onlines and iteration the max it in the gamma treshold test loop
        # Batch setting !
        if updatetype == 'batch':
            lda = Models[model].LdaModel(corpus, id2word=id2word, num_topics=K, alpha=alpha,
                                         iterations=100, eval_every=None, update_every=None, passes=50, chunksize=200000, fname=fname, heldout_corpus=heldout_corpus)
        elif updatetype == 'online':
            lda = Models[model].LdaModel(corpus, id2word=id2word, num_topics=K, alpha=alpha,
                                         iterations=100, eval_every=None, update_every=1, passes=1, chunksize=2000, fname=fname, heldout_corpus=heldout_corpus)

        if save:
            lda.expElogbeta = None
            lda.sstats = None
            lda.save(fname)
        return lda

