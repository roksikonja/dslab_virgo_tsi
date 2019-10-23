from setuptools import setup, find_packages

setup(
    name='dslab_virgo_tsi',
    version='1.0',
    url='https://github.com/roksikonja/dslab_virgo_tsi',

    author='Luka Kolar, Rok Šikonja and Lenart Treven',
    author_email='kolarl@student.ethz.ch, rsikonja@student.ethz.ch and trevenl@student.ethz.ch',
    description='ETHs Data Science Lab Project - Autumn 2019.',

    packages=find_packages(exclude=[]),
    python_requires='==3.7',
    install_requires=[
                "numpy",
                "pandas",
                "matplotlib",
                "ipykernel",
                "jupyter",
                "scipy",
                "sklearn",
                "tables",
                "numba",
                "pytables>=3.2"
        ],
)
