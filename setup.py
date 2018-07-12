from distutils.core import setup, Extension
module = Extension('mypack',extra_compile_args=['-std=c++11'], include_dirs=['/usr/local/cuda/include'],
        sources = ['Detector.cpp'],extra_objects = ['./plugin.o', './kernel.o'], extra_link_args=['-lnvinfer', '-lnvcaffe_parser', '-lcudnn'])

setup(name = 'mypack', version='1.0', description="Detector package", ext_modules = [module])

