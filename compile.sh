g++ -c test.cpp -L"/usr/local/cuda/targets/aarch64-linux/lib" -L"/usr/local/lib"  -L"/usr/local/cuda/lib64" -lnvinfer -lnvcaffe_parser -lcudnn -lcublas -lcudart_static -lnvToolsExt -lcudart -fPIC -lrt -ldl -lpthread -Wfatal-errors -Ofast  -I /usr/local/cuda/include -std=c++11 -o test.o -g


nvcc -c  kernel.cu  -std=c++11 -o kernel.o -g --compiler-options "-Wall -Wfatal-errors -Ofast -fPIC"

g++ -c plugin.cpp -L"/usr/local/cuda/targets/aarch64-linux/lib" -L"/usr/local/lib"  -L"/usr/local/cuda/lib64" -lnvinfer -lnvcaffe_parser -lcudnn -lcublas -lcudart_static -lnvToolsExt -lcudart -lrt -ldl -lpthread -Wfatal-errors -Ofast -I /usr/local/cuda/include -std=c++11 -o plugin.o  -g -fPIC

g++ kernel.o plugin.o test.o -L"/usr/local/cuda/targets/aarch64-linux/lib" -L"/usr/local/lib"  -L"/usr/local/cuda/lib64" -lnvinfer -lnvcaffe_parser -lcudnn -lcublas -lcudart_static -lnvToolsExt -lcudart -lrt -ldl -Wfatal-errors -Ofast -lpthread -lopencv_imgcodecs -lopencv_imgcodecs -lopencv_highgui -lopencv_core -g -fPIC

python setup.py build
