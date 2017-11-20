import logging
import os
import sys
import multiprocessing
 
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim import models
 
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
 
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
 
    # check and process input arguments
    if len(sys.argv) < 4:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    inp, outp1, outp2 = sys.argv[1:4]

    model = Word2Vec(LineSentence(inp), size=500, window=10, min_count=5,
                     workers=multiprocessing.cpu_count()-1)
 
    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    model.wv.save_word2vec_format("./Data/wiki_en.vec", binary=False)
    model.save("./Data/wiki_en.model.bin")
