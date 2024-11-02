from setuptools import setup, find_packages

setup(
    name="vmc_torch",
    version="0.0.1",
    description="A quantum variational Monte Carlo library using PyTorch",
    long_description=open("README.md").read(),
    url="https://github.com/sjdu10/vmc_torch",
    author="Sijing Du",
    author_email="sdu2@caltech.edu",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="tensor block sparse symmetry autoray",
    packages=find_packages(),
    install_requires=[
        "autoray",
        "numpy",
        "torch",
        "quimb @ git+https://github.com/sjdu10/quimb.git",
        "symmray @ git+https://github.com/sjdu10/symmray.git",
    ],
    include_package_data=True,
)
