rm -r sci_ztools.egg-info
rm -r build
rm -r dist
python3 setup.py sdist bdist_wheel
python3 -m twine upload dist/*
