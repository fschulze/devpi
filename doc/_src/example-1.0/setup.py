import os, sys
try:
    from setuptools import setup, Command
except ImportError:
    from distribute_setup import use_setuptools
    use_setuptools()
    from setuptools import setup, Command

def main():
    setup(
        name='example',
        description='example test project (please ignore)',
        version='1.0',
        author='Holger Krekel',
        author_email='holger at merlinux.eu',
        py_modules=["example"],
        zip_safe=False,
    )

if __name__ == '__main__':
    main()
