# codes from https://github.com/bojone/bert4keras/blob/master/examples/basic_language_model_nezha_gen_gpt.py
# refer to https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-Gen-TensorFlow for further info

import os
os.environ['TF_KERAS'] = '1'
import numpy as np
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import AutoRegressiveDecoder
from bert4keras.snippets import uniout

relative = lambda x: os.path.join(os.path.dirname(os.path.abspath(__file__)), x)
config_path = relative('config.json')
checkpoint_path = relative('gpt.ckpt')
dict_path = relative('vocab.txt')

tokenizer = Tokenizer(dict_path, do_lower_case=True)

model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    segment_vocab_size=0,
    application='lm',
)


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


article_completion = ArticleCompletion(
    start_id=None,
    end_id=511,  # 511是中文句号
    maxlen=64,
    minlen=32
)
