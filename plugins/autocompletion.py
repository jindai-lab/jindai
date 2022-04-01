"""自动生成文本
"""

# codes from https://github.com/bojone/bert4keras/blob/master/examples/basic_language_model_nezha_gen_gpt.py
# refer to https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-Gen-TensorFlow for further info

import os

os.environ['TF_KERAS'] = '1'
import numpy as np
from jindai import Plugin, expand_path
from jindai.helpers import safe_import
from jindai.models import Paragraph
from jindai.pipeline import DataSourceStage


class AutoCompletionPlugin(Plugin):
    
    def __init__(me, app, **config) -> None:
        super().__init__(app, **config)
        
        safe_import('bert4keras')
        from bert4keras.models import build_transformer_model
        from bert4keras.snippets import AutoRegressiveDecoder
        from bert4keras.tokenizers import Tokenizer
        
        me.model = None
        
        class ArticleCompletion(AutoRegressiveDecoder):
            """基于随机采样的文章续写
            """
            
            @AutoRegressiveDecoder.wraps(default_rtype='probas')
            def predict(self, inputs, output_ids, states):
                token_ids = np.concatenate([inputs[0], output_ids], 1)
                return me.last_token(me.model).predict(token_ids)

            def generate(self, text, n=1, topp=0.95):
                token_ids = self.tokenizer.encode(text)[0][:-1]
                results = self.random_sample([token_ids], n, topp=topp)
                return [text + self.tokenizer.decode(ids) for ids in results]

        class AutoCompletionDataSource(DataSourceStage):
            """从输入的提示文本自动生成
            """

            class _Implementation(DataSourceStage._Implementation):

                def __init__(self, dataset_name, prompts, n=5, topp=0.9):
                    """
                    Args:
                        dataset_name (DATASET): 数据集名称
                        prompts (str): 提示文本，一行一个
                        n (int): 针对每个提示文本生成的样本数量
                        topp (float): 概率阈值
                    """
                    super().__init__()
                
                    relative = lambda x: expand_path(f'models_data/autocompletion/{x}')
                    config_path = relative('config.json')
                    checkpoint_path = relative('gpt.ckpt')
                    dict_path = relative('vocab.txt')

                    self.article_completion = ArticleCompletion(
                        start_id=None,
                        end_id=511,
                        maxlen=64,
                        minlen=32
                    )
                    me.model = build_transformer_model(
                        config_path=config_path,
                        checkpoint_path=checkpoint_path,
                        segment_vocab_size=0,
                        application='lm',
                    )
                    self.article_completion.tokenizer = Tokenizer(dict_path, do_lower_case=True)

                    self.dataset_name = dataset_name
                    self.lang = 'chs'
                    self.prompts = prompts.split('\n')
                    self.n = n
                    self.topp = topp

                def count(self):
                    return self.n * len(self.prompts)

                def fetch(self):
                    for prompt in self.prompts:
                        for r in self.article_completion.generate(prompt, self.n, self.topp):
                            yield Paragraph(
                                content=r,
                                lang='chs',
                                dataset=self.dataset_name,
                                source={'text': prompt}
                            )

        me.register_pipelines(locals())
