from setuptools import find_packages,setup
setup(
    name='Student_Analysis',
    version='0.0.1',
    author='sourabh kumar',
    author_email='sourabhkumar9221@gmail.com',
    install_requires=["scikit-learn","pandas","numpy"],
    packages=find_packages()
)