from setuptools import setup
from codecs import open
from os import path
from os.path import splitext
import glob

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


def get_py_modules(excludes=['__init__.py', 'setup.py']):
    return [
        splitext(p)[0] for p in sorted(glob.glob('*.py')) if p not in excludes
    ]


setup(
    name='text_word_embed',
    version='0.1.0',
    description='Word embedding examples',
    long_description=long_description,
    author='Grzegorz Zycinski',
    author_email='g.zycinski@gmail.com',
    url='http://github.com/gregor8003/text_word_embed/',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Text Processing :: General',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='text processing word embeddings neural networks',
    py_modules=get_py_modules(),
    zip_safe=False
)
