import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = "0.1.0"
PACKAGE_NAME = "pix2pix"
AUTHOR = "Sambhav"
AUTHOR_EMAIL = "sambhav300899@email.com"
URL = "https://github.com/Sambhav300899/pix2pix-image-denoising.git"

LICENSE = "Apache License 2.0"
DESCRIPTION = "Pix2Pix implemented in PyTorch"
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = open("requirements.txt", "r").readlines()

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    package_dir={"": "src"},
    packages=find_packages("src"),
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    author=AUTHOR,
    license=LICENSE,
    author_email=AUTHOR_EMAIL,
    url=URL,
    install_requires=INSTALL_REQUIRES,
)
