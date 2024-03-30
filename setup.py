from setuptools import setup, find_packages

setup(
    name='timbre-trap',
    url='https://github.com/sony/timbre-trap',
    author='Frank Cwitkowitz',
    author_email='fcwitkow@ur.rochester.edu',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[],
    version='0.1.0',
    license='MIT',
    description='Code for Timbre-Trap framework',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown'
)