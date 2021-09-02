"""自动生成文本
"""

# codes from https://github.com/bojone/bert4keras/blob/master/examples/basic_language_model_nezha_gen_gpt.py
# refer to https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-Gen-TensorFlow for further info

from models import Paragraph
from datasource import DataSource
import config
import os
os.environ['TF_KERAS'] = '1'
import numpy as np
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import AutoRegressiveDecoder
from bert4keras.snippets import uniout


class ArticleCompletion(AutoRegressiveDecoder):
    """基于随机采样的文章续写
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids = np.concatenate([inputs[0], output_ids], 1)
        return self.last_token(model).predict(token_ids)

    def generate(self, text, n=1, topp=0.95):
        token_ids = tokenizer.encode(text)[0][:-1]
        results = self.random_sample([token_ids], n, topp=topp)
        return [text + tokenizer.decode(ids) for ids in results]


model = None
relative = lambda x: os.path.join(config.rootpath, 'models_data/autocompletion', x)
config_path = relative('config.json')
checkpoint_path = relative('gpt.ckpt')
dict_path = relative('vocab.txt')
tokenizer = Tokenizer(dict_path, do_lower_case=True)
article_completion = ArticleCompletion(
    start_id=None,
    end_id=511,
    maxlen=64,
    minlen=32
)


class AutoCompletionDataSource(DataSource):
    """从输入的提示文本自动生成
    """

    def __init__(self, collection_name, prompts, n=5, topp=0.9):
        """
        Args:
            collection_name (str): 集合名称
            prompts (str): 提示文本，一行一个
            n (int): 针对每个提示文本生成的样本数量
            topp (float): 概率阈值
        """
        global model
        model = build_transformer_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            segment_vocab_size=0,
            application='lm',
        )
        self.generator = article_completion
        self.collection_name = collection_name
        self.lang = 'chs'
        self.prompts = prompts.split('\n')
        self.n = n
        self.topp = topp

    def count(self):
        return self.n * len(self.prompts)

    def fetch(self):
        for prompt in self.prompts:
            for r in self.generator.generate(prompt, self.n, self.topp):
                yield Paragraph(
                    content=r,
                    lang='chs',
                    collection=self.collection_name,
                    source={'text': prompt}
                )
