from typing import Iterable
from models import Paragraph
    
class DataSource:

    def fetch(self) -> Iterable[Paragraph]:
        return []
