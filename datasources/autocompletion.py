"""自动生成文本
"""
from models import Paragraph
from datasource import DataSource


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
        from articlecompletion import article_completion
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
                    pdffile=prompt
                )
