#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

import platform
from distutils.sysconfig import get_config_var
from distutils.version import LooseVersion


# Package meta-data.
NAME = "py-mint"
DESCRIPTION = (
    "PyMint (Python-based Model INTerpretations) is a user-friendly python package"
    + " for computing and plotting machine learning interpretation output."
)
URL = "https://github.com/monte-flora/py-mint/"
EMAIL = "monte.flora@noaa.gov"
AUTHOR = "Montgomery Flora"
REQUIRES_PYTHON = ">=3.8.0"
VERSION = "0.2.6"

# What packages are required for this module to be executed?
REQUIRED = [
    "numpy",
    "pandas",
    "scikit-learn>=1.0.0",
    "matplotlib",
    "shap>=0.30.0",
    "xarray>=0.16.0",
    "tqdm", 
    "statsmodels",
    "seaborn>=0.11.0",
]

# What packages are optional?
EXTRAS = {
    "interactive": ["jupyter"],
}

if sys.platform == "darwin":
    if "MACOSX_DEPLOYMENT_TARGET" not in os.environ:
        current_system = LooseVersion(platform.mac_ver()[0])
        python_target = LooseVersion(get_config_var("MACOSX_DEPLOYMENT_TARGET"))
        if python_target < "10.9" and current_system >= "10.9":
            os.environ["MACOSX_DEPLOYMENT_TARGET"] = "10.9"


# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__init__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload --repository pypi dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(about["__version__"]))
        os.system("git push --tags")

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=[
                "pymint",
                "pymint.common", 
                "pymint.main", 
                "pymint.main.PermutationImportance",
                "pymint.plot", 
                ],
    package_data = {'pymint' : ['common/data/*', 'common/models/*']},
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    setup_requires=["wheel"],
    include_package_data=True,
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    # $ setup.py publish support.
    cmdclass={
        "upload": UploadCommand,
    },
)
