"""Word extraction based on PMILR
@chs 基于信息熵的词汇抽取词汇抽取"""
import re

import numpy as np
from jindai.models import Paragraph

from .basics import NgramCounter


class PMILREntropyWordFetcher(NgramCounter):
    """Word extraction based on PMILR
    @chs 基于信息熵的词汇抽取
    """
    re_stopwords = re.compile(r'[a-zA-Z0-9一二三四五六七八九十]')

    def __init__(self, word_length=4, min_pmi=2, min_lr_ent=1, min_freq=1e-5, **_):
        """
        Args:
            word_length (int, optional): Max word length
                @chs 抽取的最大词长度。
            min_pmi (float, optional): Min PMI thresholdd
                @chs 最小互信息熵阈值。
            min_lr_ent (float, optional): Min L-R Entropy, 0 to skip
                @chs 最小左右信息熵阈值，若为0则将跳过。
            min_freq (float, optional): Min frequency
                @chs 最小词频。
        """
        self.min_pmi = min_pmi
        self.min_lr_ent = min_lr_ent
        self.min_freq = min_freq
        self.word_length = word_length
        self.text_length = {}
        self.result = {}
        super().__init__(word_length, self.min_lr_ent > 0)

    def resolve(self, paragraph: Paragraph):
        self.text_length[paragraph.id] = len(paragraph.content)
        super().resolve(paragraph)

    def summarize(self, _):
        self.text_length = sum(self.text_length.values())
        result = self.pmient()
        if self.min_lr_ent > 0:
            result = self.ent_lr()
        return sorted([(w, self.ngrams[w]) for w in result], key=lambda x: x[1], reverse=True)

    def pmient(self):
        """Calculating PMI Entropy"""
        candidates = []
        min_p = 2 ** self.min_pmi / self.text_length
        min_freq = int(self.min_freq * self.text_length)
        self.logger('generating pmient words...')

        for word in list(self.ngrams):
            if 1 < len(word) <= self.word_length and self.ngrams[word] > min_freq \
                    and not PMILREntropyWordFetcher.re_stopwords.search(word):
                px_py = min([self.ngrams.get(word[:i], 1) * self.ngrams.get(word[i:], 1)
                             for i in range(1, len(word))])
                mpi = self.ngrams[word] / px_py
                if mpi > min_p:
                    candidates.append(word)

        self.result['pmi_words'] = candidates
        return candidates

    def ent_lr(self):
        """Left Right Entropy"""
        def calculate_entropy(char_dic):
            if not char_dic:
                return 0
            freq = np.array(list(char_dic.values()), dtype='f')
            freq /= freq.sum()
            freq *= -np.log2(freq)
            entropy = freq.sum()
            return entropy

        final_words = []
        if 'pmi_words' not in self.result:
            self.pmient()

        self.logger('filtering lr...')
        for word in self.result['pmi_words']:
            left_entropy = calculate_entropy(self.ngrams_lefts.get(word, {}))
            right_entropy = calculate_entropy(self.ngrams_rights.get(word, {}))

            if min(right_entropy, left_entropy) > self.min_lr_ent:
                final_words.append(word)

        self.result['lr_words'] = final_words
        return final_words
