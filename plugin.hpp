#ifndef __PLUGIN_LAYER_H_
#define __PLUGIN_LAYER_H_

#include "NvInferPlugin.h"
#include "NvCaffeParser.h"
#include <cassert>
#include <cudnn.h>
#include <iostream>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>

//#define RESIZE_W 608
//#define RESIZE_H 352
#define RESIZE_W 416
#define RESIZE_H 288
#define CHANNEL 3

#define ORIGIN_W 640
#define ORIGIN_H 360

#define REORG_STRIDE 2
#define CLASS_NUM 1
#define BOX_NUM 10
#define MAX_BATCH_SIZE 4
#define BATCH_SIZE 2

#define THRESH_NMS 0.4
#define THRESH_PROB 0.005

#define KERNEL_SIZE 1

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;
using std::vector;
using std::cout;
using std::endl;
using std::string;

//#define DEBUG 1
//#define DEBUG_PREPROCESS 1
//#define DEBUG_POSTPROCESS 1
//#define DEBUG_LEAKYRELU 1
//#define DEBUG_POSTPROCESS_INPUT 1
//#define DEBUG_OUTPUTLAYER 1
/*#define DEBUG_DEPTHWISE 1*/

#define CHECK(status)                                                                                           \
    {                                                                                                                           \
        if (status != 0)                                                                                                \
        {                                                                                                                               \
            std::cout << "Cuda failure: " << cudaGetErrorString(status) \
                      << " at line " << __LINE__                                                        \
                      << std::endl;                                                                     \
            abort();                                                                                                    \
        }                                                                                                                               \
    }

struct sbbox {
    float x, y, w, h, score;
    sbbox(float x_, float y_, float w_, float h_, float score_):
        x(x_), y(y_), w(w_), h(h_), score(score_){};
};

class PreprocessLayer: public IPlugin{
    public:
        PreprocessLayer(){};
        inline int getNbOutputs() const override {return 1;}
        Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;
        
        int initialize() override;
        inline void terminate() override {};
        inline size_t getWorkspaceSize(int) const override {return 0;}
        int enqueue(int batchSize, const void*const *inputs, void **outputs, void *, cudaStream_t stream) override;
        
        size_t getSerializationSize() override;
        void serialize(void *buffer) override;
        
        void configure(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override;
    protected:
/*        vector<int> old_steps_;*/
        //vector<int> new_steps_;
        //vector<int> permute_order_;
        //int num_axes;
        //vector<float> mean_values_;
        /*float scale_;*/
};

class ReorgLayer: public IPlugin {
    public:
        ReorgLayer(){};
        Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;
            
        int initialize() override {return 0;};
        int enqueue(int batchSize, const void *const* inputs, void **output, void *, cudaStream_t stream) override;
            
        size_t getSerializationSize() override {return 0;};
        void serialize(void *buffer) override {return ;};
        void configure(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override ;
        inline int getNbOutputs() const override {return 1;}
        inline size_t getWorkspaceSize(int) const override {return 0;}
        inline void terminate() override {};
};

class PostProcessLayer: public IPlugin {
    public: 
        PostProcessLayer() {};
        Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;
            
        int initialize() override {return 0;};
        int enqueue(int batchSize, const void *const* inputs, void **output, void *, cudaStream_t stream) override;
            
        size_t getSerializationSize() override {return 0;};
        void serialize(void *buffer) override {return; };
        void configure(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override ;
        inline int getNbOutputs() const override {return 1;}
        inline size_t getWorkspaceSize(int) const override {return 0;}
        inline void terminate() override {};
};

class LeakyReLULayer: public IPlugin {
    public: 
        LeakyReLULayer() {
#ifdef DEBUG 
            cout <<"create leakyRelu layer"<<endl;
#endif
        };
        LeakyReLULayer(const void *data, size_t length) ;
        Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;
            
        int initialize() override {return 0;};
        int enqueue(int batchSize, const void *const* inputs, void **output, void *, cudaStream_t stream) override;
            
        size_t getSerializationSize() override ;
        void serialize(void *buffer) override;
        void configure(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override  ;
        inline int getNbOutputs() const override {return 1;}
        inline size_t getWorkspaceSize(int) const override {return 0;}
        inline void terminate() override {};
    private :
        template<typename T> void write(char*& buffer, const T& val){
	    	*reinterpret_cast<T*>(buffer) = val;
	    	buffer += sizeof(T);
	    }

	    template<typename T> T read(const char*& buffer){
	    	T val = *reinterpret_cast<const T*>(buffer);
	    	buffer += sizeof(T);
	    	return val;
	    }

	    Weights copyToDevice(const void* hostData, size_t count){
	    	void* deviceData;
	    	CHECK(cudaMalloc(&deviceData, count * sizeof(float)));
	    	CHECK(cudaMemcpy(deviceData, hostData, count * sizeof(float), cudaMemcpyHostToDevice));
	    	return Weights{ DataType::kFLOAT, deviceData, int64_t(count) };
	    }

	    void serializeFromDevice(char*& hostBuffer, Weights deviceWeights){		
	    	cudaMemcpy(hostBuffer, deviceWeights.values, deviceWeights.count * sizeof(float), cudaMemcpyDeviceToHost);
	    	hostBuffer += deviceWeights.count * sizeof(float);
	    }

	    Weights deserializeToDevice(const char*& hostBuffer, size_t count){
	    	Weights w = copyToDevice(hostBuffer, count);
	    	hostBuffer += count * sizeof(float);
	    	return w;	
	    }
        int count;
};


class OutputLayer: public IPlugin {
    public:
        OutputLayer() {
#ifdef DEBUG
            cout <<"create output layer"<<endl;
#endif
        };
        Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;
            
        int initialize() override {return 0;};
        int enqueue(int batchSize, const void *const* inputs, void **output, void *, cudaStream_t stream) override;
            
        size_t getSerializationSize() override {return 0;};
        void serialize(void *buffer) override {return; };
        void configure(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override ;
        inline int getNbOutputs() const override {return 1;}
        inline size_t getWorkspaceSize(int) const override {return 0;}
        inline void terminate() override {};
};

class DepthwiseConvLayer_s1 : public IPlugin{
    public:
        DepthwiseConvLayer_s1(const nvinfer1::Weights* weights, int nbWeights) ;
        DepthwiseConvLayer_s1(const void *data, size_t length);
        Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;
            
        int initialize() override {return 0;};
        int enqueue(int batchSize, const void *const* inputs, void **output, void *, cudaStream_t stream) override ;
            
        size_t getSerializationSize() override;
        void serialize(void *buffer) override ;
        void configure(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override;
        inline int getNbOutputs() const override {return 1;}
        inline size_t getWorkspaceSize(int) const override {return 0;}
        inline void terminate() override {};
        ~DepthwiseConvLayer_s1() {
            cudaFree(const_cast<void*>(kernelWeight.values));
            cudaFree(const_cast<void*>(biasWeight.values));
        }
    private:
        template<typename T> void write(char*& buffer, const T& val){
	    	*reinterpret_cast<T*>(buffer) = val;
	    	buffer += sizeof(T);
	    }

	    template<typename T> T read(const char*& buffer){
	    	T val = *reinterpret_cast<const T*>(buffer);
	    	buffer += sizeof(T);
	    	return val;
	    }

	    Weights copyToDevice(const void* hostData, size_t count){
	    	void* deviceData;
	    	CHECK(cudaMalloc(&deviceData, count * sizeof(float)));
	    	CHECK(cudaMemcpy(deviceData, hostData, count * sizeof(float), cudaMemcpyHostToDevice));
	    	return Weights{ DataType::kFLOAT, deviceData, int64_t(count) };
	    }

	    void serializeFromDevice(char*& hostBuffer, Weights deviceWeights){		
	    	cudaMemcpy(hostBuffer, deviceWeights.values, deviceWeights.count * sizeof(float), cudaMemcpyDeviceToHost);
	    	hostBuffer += deviceWeights.count * sizeof(float);
	    }

	    Weights deserializeToDevice(const char*& hostBuffer, size_t count){
	    	Weights w = copyToDevice(hostBuffer, count);
	    	hostBuffer += count * sizeof(float);
	    	return w;	
	    }
        int height;
        int width;
        int channels;
        const static int stride =1;
        Weights kernelWeight, biasWeight;
};

class DepthwiseConvLayer_s2 : public IPlugin{
    public:
        DepthwiseConvLayer_s2(const nvinfer1::Weights* weights, int nbWeights) ;
        DepthwiseConvLayer_s2(const void *data, size_t length);
        Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;
            
        int initialize() override {return 0;};
        int enqueue(int batchSize, const void *const* inputs, void **output, void *, cudaStream_t stream) override ;
            
        size_t getSerializationSize() override;
        void serialize(void *buffer) override ;
        void configure(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override;
        inline int getNbOutputs() const override {return 1;}
        inline size_t getWorkspaceSize(int) const override {return 0;}
        inline void terminate() override {};
        ~DepthwiseConvLayer_s2() {
            cudaFree(const_cast<void*>(kernelWeight.values));
            cudaFree(const_cast<void*>(biasWeight.values));
        }
    private:
        template<typename T> void write(char*& buffer, const T& val){
	    	*reinterpret_cast<T*>(buffer) = val;
	    	buffer += sizeof(T);
	    }

	    template<typename T> T read(const char*& buffer){
	    	T val = *reinterpret_cast<const T*>(buffer);
	    	buffer += sizeof(T);
	    	return val;
	    }

	    Weights copyToDevice(const void* hostData, size_t count){
	    	void* deviceData;
	    	CHECK(cudaMalloc(&deviceData, count * sizeof(float)));
	    	CHECK(cudaMemcpy(deviceData, hostData, count * sizeof(float), cudaMemcpyHostToDevice));
	    	return Weights{ DataType::kFLOAT, deviceData, int64_t(count) };
	    }

	    void serializeFromDevice(char*& hostBuffer, Weights deviceWeights){		
	    	cudaMemcpy(hostBuffer, deviceWeights.values, deviceWeights.count * sizeof(float), cudaMemcpyDeviceToHost);
	    	hostBuffer += deviceWeights.count * sizeof(float);
	    }

	    Weights deserializeToDevice(const char*& hostBuffer, size_t count){
	    	Weights w = copyToDevice(hostBuffer, count);
	    	hostBuffer += count * sizeof(float);
	    	return w;	
	    }
        int height;
        int width;
        int channels;
        const static int stride = 2;
        Weights kernelWeight, biasWeight;
};

#ifdef DEBUG
class TestLayer: public IPlugin {
    public:
        TestLayer(const nvinfer1::Weights* weights, int nbWeights) ;
        TestLayer(const void *data, size_t length);
        Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;
            
        int initialize() override {return 0;};
        int enqueue(int batchSize, const void *const* inputs, void **output, void *, cudaStream_t stream) override;
            
        size_t getSerializationSize() override {return 0;};
        void serialize(void *buffer) override {return ;};
        void configure(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override ;
        inline int getNbOutputs() const override {return 1;}
        inline size_t getWorkspaceSize(int) const override {return 0;}
        inline void terminate() override {};
    protected:
        int count;
};
#endif


class PluginFactory:public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory{
    public:
        typedef std::pair<std::string, std::unique_ptr<DepthwiseConvLayer_s1> >DepthwiseLayer_s1;
        typedef std::pair<std::string, std::unique_ptr<DepthwiseConvLayer_s2> >DepthwiseLayer_s2;
        virtual nvinfer1::IPlugin* createPlugin(const char * layerName, const nvinfer1::Weights* weights, int nbWeights) override;
        IPlugin* createPlugin(const char* layerName, const void *serialData, size_t serialLength) override;
            
        bool isPlugin(const char* name)override;
        void destroyPlugin();
        enum LayerType{
            PREPROCESSLAYER = 0,
            POSTPROCESSLAYER,
            REORGLAYER,
            DEPTHWISELAYER_S1,
            DEPTHWISELAYER_S2,
            OUTPUTLAYER,
            LEAKYRELULAYER,
#ifdef DEBUG
            TESTLAYER,
#endif
            NOT_SUPPORTED
        };
        LayerType getLayerType(const char *name);
        
    private:
        std::vector<string> dep_conv_s1_layers_name{"conv2_1/dw", "conv3_1/dw", "conv4_1/dw", "conv5_1/dw", "conv5_2/dw", "conv5_3/dw", "conv5_4/dw", "conv5_5/dw", "conv6/dw"};
        std::vector<string> dep_conv_s2_layers_name{"conv2_2/dw", "conv3_2/dw", "conv4_2/dw", "conv5_6/dw"};
        std::vector<DepthwiseLayer_s1> depLayers_s1;
        std::vector<DepthwiseLayer_s2> depLayers_s2;
        std::vector<std::unique_ptr<LeakyReLULayer> > leakyreluvec;
        std::unique_ptr<PreprocessLayer> preprocessLayer{nullptr};
        std::unique_ptr<ReorgLayer> reorgLayer{nullptr};
        std::unique_ptr<PostProcessLayer> postProcessLayer{nullptr};
        std::unique_ptr<OutputLayer> outputLayer{nullptr};
        std::unique_ptr<LeakyReLULayer> pLeakyReluLayer{nullptr};
        std::unique_ptr<DepthwiseConvLayer_s1> dLayer_s1{nullptr};
        std::unique_ptr<DepthwiseConvLayer_s2> dLayer_s2{nullptr};
#ifdef DEBUG
        std::unique_ptr<TestLayer> testLayer{nullptr};
        std::vector<string> test_layers_name{"sda"};
#endif
};

#endif
