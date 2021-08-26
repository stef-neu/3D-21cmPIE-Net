from setuptools import find_packages, setup

setup(
    name="3D-21cmPIE-Net",
    version="0.1.0",
    description="3D CNN and tools for parameter inference from 21 cm maps",
    author="Steffen Neutsch",
    author_email="steffen.neutsch@hs.uni-hamburg.de",
    packages=find_packages(),
    license="BSD",
    install_requires = [
        "numpy",
        "tensorflow>2.0",
        "matplotlib",
        "py21cmfast>=3.1",
    ],
    python_requires='>=3.6',
)
