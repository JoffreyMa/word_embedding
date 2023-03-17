from setuptools import setup, find_packages

setup(
    name="word_embedding",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # Add your package dependencies here, e.g.,
        # "numpy",
        # "pandas",
    ],
    author="Joffrey Ma",
    author_email="ma.joffrey@gmail.com",
    description="Study of word embedding",
    url="https://github.com/JoffreyMa/word_embedding",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
