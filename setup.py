import re
from setuptools import find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools import setup

from cdd.info import info

class PackageDevelop(develop):
    def run(self):
        develop.run(self)


class PackageInstall(install):
    def run(self):
        # install everything else
        install.run(self)


# use requirements.txt as deps list
with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

# get readme
with open('README.md', 'r') as f:
    readme = f.read()

test_deps = ["green>=2.5.0,<3", "coverage", "mypy"]

setup(name='cluling-arabic-nlp',
      packages=find_packages(),
      #packages=["cdd"],
      scripts=['bin/train-transformer'],
      version=info.version,
      keywords=['nlp', 'cluling', 'arabic'],
      description=info.description,
      long_description=readme,
      url=info.repo,
      download_url=info.download_url,
      author=info.author,
      author_email=info.contact,
      license=info.license,
      install_requires=required,
      cmdclass={
        'install': PackageInstall,
        'develop': PackageDevelop,
      },
      classifiers=[
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3"
      ],
      python_requires=">=3.8",
      tests_require=test_deps,
      extras_require={
        'test': test_deps,
        'all': test_deps + required
      },
      include_package_data=True,
      zip_safe=False)

