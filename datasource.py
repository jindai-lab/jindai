from typing import Iterable
from models import Paragraph
    
class DataSource:

    def __init__(self) -> None:
        self.logger = print

    def fetch(self) -> Iterable[Paragraph]:
        return []
