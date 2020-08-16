# Publish the package to TestPyPI 

import os
os.system('twine upload --repository-url https://test.pypi.org/legacy/ dist/*')
