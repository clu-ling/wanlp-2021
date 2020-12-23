from dataclasses import dataclass

@dataclass(frozen=True)
class AppInfo:
    """
    General information about the application.
    """
    version: str = "0.1"
    description: str = "Classifiers and utilities for Arabic NLP."
    author: str = "myedibleenso"
    contact: str = "gus@parsertongue.org"
    repo: str = "https://github.com/clu-ling/arabic-nlp"
    license: str = "TBD"
    
    @property
    def download_url(self) -> str: 
      return f"{self.repo}/archive/v{self.version}.zip"
    

info = AppInfo()
