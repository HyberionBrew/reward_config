from setuptools import setup, find_packages

setup(
    name='f110_orl_dataset',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
        'f110-gym',
        'zarr',
    ],
)