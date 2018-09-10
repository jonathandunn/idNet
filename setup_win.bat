python setup.py sdist
python setup.py bdist_wheel
@RD /S /Q "idNet.egg-info"
@RD /S /Q "build"