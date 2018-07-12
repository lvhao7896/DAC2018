#ifndef _DETECTOR_H_
#define _DETECTOR_H_
#include <algorithm>
#include "plugin.hpp"
#include "NvInferPlugin.h"
#include "NvCaffeParser.h"
#include <cudnn.h>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace nvinfer1;
using namespace nvcaffeparser1;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB = "region_output";

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

class Logger:public ILogger{
    void log(Severity serverity, const char *msg) override {
        if (serverity != Severity::kINFO) std::cout <<msg << std::endl;
    }
};

class Profiler : public IProfiler {
    public:
    typedef std::pair<std::string, float> Record;
    std::vector<Record> mProfile;

    virtual void reportLayerTime(const char* layerName, float ms)
    {
        auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record& r){ return r.first == layerName; });

        if (record == mProfile.end()) mProfile.push_back(std::make_pair(layerName, ms));
        else record->second += ms;
    }

    void printLayerTimes(const int TIMING_ITERATIONS)
    {
        float totalTime = 0;
        for (size_t i = 0; i < mProfile.size(); i++)
        {
            printf("%-40.40s %4.3fms(%4.3f/%d)\n", mProfile[i].first.c_str(), mProfile[i].second / TIMING_ITERATIONS, mProfile[i].second, TIMING_ITERATIONS);
            totalTime += mProfile[i].second;
        }
        printf("Time over all layers: %4.3fms(%4.3f/%d)\n", totalTime / TIMING_ITERATIONS, totalTime, TIMING_ITERATIONS);
    }
};


class DetectNet {
    public:
        void loadNet(const std::string &planFile) {
            cout <<"Loading plan file "<<planFile<<endl;
            infer = createInferRuntime(gLogger);
            std::ifstream ifile(planFile);
            assert(ifile.is_open());
            std::stringstream serial_data;
            serial_data <<ifile.rdbuf();
            ifile.close();
            serial_data.seekg(0, std::ios::end);
            const int model_size = serial_data.tellg();
            serial_data.seekg(0, std::ios::beg);
            void *modelMem = malloc(model_size);
            assert(modelMem != nullptr);
            serial_data.read((char*)modelMem, model_size);
            engine = infer->deserializeCudaEngine(modelMem, model_size, &pluginFactory);
            context = engine->createExecutionContext();
            free(modelMem);
            DimsCHW dimsData = getTensorDims(INPUT_BLOB_NAME);
            DimsCHW dimsout = getTensorDims(OUTPUT_BLOB);
            input_blob = allocateMemory(dimsData, (char*)"input_blob");
            output_blob = allocateMemory(dimsout, (char*)"region_output");
            CHECK(cudaStreamCreate(&stream));
        }   
            
        void inference(float* img, int batch_size, float*result,int data_size) {
            void *buffers[] = {input_blob, output_blob};
            assert(engine->getNbBindings() == 2);
            //DimsCHW dimsData = getTensorDims(INPUT_BLOB_NAME);
            //DimsCHW dimsout = getTensorDims(OUTPUT_BLOB);
            //int outputsz = dimsout.c() * dimsout.w() * dimsout.h();
            CHECK(cudaMemcpy(input_blob, img, data_size*sizeof(float), cudaMemcpyHostToDevice));
            //CHECK(cudaMemcpyAsync(input_blob, img, data_size*sizeof(float), cudaMemcpyHostToDevice, stream));
            //std::cout <<"infering ..."<<std::endl;
            context->execute(batch_size, buffers);
            //context->enqueue(batch_size, buffers, stream, nullptr);
            //CHECK(cudaMemcpyAsync(result, buffers[engine->getBindingIndex(OUTPUT_BLOB)], 7*batch_size*sizeof(float), cudaMemcpyDeviceToHost, stream));
            CHECK(cudaMemcpy(result, buffers[engine->getBindingIndex(OUTPUT_BLOB)], 4*batch_size*sizeof(float), cudaMemcpyDeviceToHost));
            //cudaStreamSynchronize(stream);
        }

        DimsCHW getTensorDims(const char *name) {
            for (int b = 0; b < engine->getNbBindings(); b++) {
                if (!strcmp(name, engine->getBindingName(b))) {
                    return static_cast<DimsCHW&&>(engine->getBindingDimensions(b));
                }
            }
            return DimsCHW{0,0,0};
        }
            
        ~DetectNet() {
            pluginFactory.destroyPlugin();
            engine->destroy();
            infer->destroy();
            cudaFree(input_blob);
            cudaFree(output_blob);
            cudaStreamDestroy(stream);
            context->destroy();
        }
    private:
        float* allocateMemory(DimsCHW dims, char* info)
        {
            float* ptr = nullptr;
            size_t size;
            size = MAX_BATCH_SIZE * dims.c() * dims.h() * dims.w();
            //std::cout << "Allocate memory: " << info <<" size "<<size<< std::endl;
            CHECK(cudaMallocManaged(&ptr, size*sizeof(float)));
            return ptr;
        }
        PluginFactory pluginFactory;
        IRuntime* infer;
        ICudaEngine* engine;
        Logger gLogger;
        float* input_blob;
        float* output_blob;
        cudaStream_t stream;
        IExecutionContext* context;
};

#endif
