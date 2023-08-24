from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    readme_text = fh.read()

setup(
    name="dfpl",
    version="1.2",
    author="Jana Schor, Patrick Scheibe, Matthias Bernt",
    author_email="jana.schor@ufz.de",
    packages=find_packages(include=["dfpl", "dfpl.*"]),
    long_description=readme_text,
    long_description_content_type="text/markdown",
    url="https://github.com/yigbt/deepFPlearn",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    # all packages need for the final usage
    # for additional packages during development, use requirements.txt
    install_requires=[
        "jsonpickle~=2.1.0",
        "matplotlib==3.5.1",
        "numpy==1.22.0",
        "pandas==1.4.2",
        "rdkit-pypi==2022.03.1",
        "scikit-learn==1.0.2",
        "keras==2.9.0",
        "tensorflow-gpu==2.9.3",
        "wandb~=0.12.0",
        "chemprop @ git+https://github.com/soulios/chemprop.git@044f9d0dbd52741d66886a2f02045bd608986ce4"
    ],
    entry_points={"console_scripts": ["dfpl=dfpl.__main__:main"]},
)
