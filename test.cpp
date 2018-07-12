#include "test.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <sstream>
using namespace cv;


//const char *model = "./yolov2_class1_kmeans_608.prototxt";
//const char *weight = "./yolov2_class1_kmeans_608.caffemodel";
const char *model = "yolov2_class1_kmeans_416_10box.prototxt";
const char *weight = "./yolov2_class1_kmeans_416_10boxes_90000.caffemodel";
//const char *weight = "./yolov2_demo.caffemodel";
//const char *model = "../model_zoo/yolov2_tucker_layer_conv21_1_2/yolov2_tucker_layer_conv21_1_2.prototxt";
//const char *weight = "../model_zoo/yolov2_tucker_layer_conv21_1_2/yolov2_tucker_layer_conv21_1_2.caffemodel";

//const char *model = "../model_zoo/yolov2_tucker_layer_conv21_1_1/yolov2_tucker_layer_conv21_1_1.prototxt";
//const char *weight = "../model_zoo/yolov2_tucker_layer_conv21_1_1/yolov2_tucker_layer_conv21_1_1.caffemodel";
//const char *model = "../model_zoo/yolov2_tucker_layer_all_1_2/yolov2_tucker_layer_all_1_2.prototxt";
//const char *weight = "../model_zoo/yolov2_tucker_layer_all_1_2/yolov2_tucker_all_1_2.caffemodel";
//const char *model = "../model_zoo/yolov2_tucker_layer_all_1_1/yolov2_tucker_layer_all_1_1.prototxt";
//const char *weight = "../model_zoo/yolov2_tucker_layer_all_1_1/yolov2_tucker_layer_all_1_1.caffemodel";
//const char * pic_path = "/home/lvhao/dji-sdc/data_training/360(71)/0511.jpg";
const char *pic_path = "./1.jpg";
//const char* pic_path = "/home/nvidia/mydisk/yolo_tensorrt/images/1.jpg";
const char * REGION_OUTPUT = "region_output";
const char *INPUT_BLOB_NAME = "data";
static const int TIMING_ITERATIONS = 1000;
const char * SERIALIZE_FILE = "tensornet.plan";


void TensorNet::caffeToTRTModel(const std::string& deployFile,
                                const std::string& modelFile,
                                const std::vector<std::string>& outputs,
                                unsigned int maxBatchSize)
{
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();

    ICaffeParser* parser = createCaffeParser();
    parser->setPluginFactory(&pluginFactory);

    bool fp16 = builder->platformHasFastFp16();
    const IBlobNameToTensor *blobNameToTensor =	parser->parse(deployFile.c_str(),
                                                              modelFile.c_str(),
                                                              *network,
                                                              fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT);

    assert(blobNameToTensor != nullptr);
    for (auto& s : outputs) network->markOutput(*blobNameToTensor->find(s.c_str()));

    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(4 << 30);
    builder->setHalf2Mode(fp16);


    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    network->destroy();
    parser->destroy();

    cout <<"Serializing to "<<SERIALIZE_FILE<<endl;
    gieModelStream = engine->serialize();
/*    std::ofstream outFile;*/
    //outFile.open(SERIALIZE_FILE);
    //std::ostream outstream;
    //outstream.write((const char*)gieModelStream->data(), gieModelStream->size());
    //outFile << outstream.rdbuf();
    //gieModelStream->destroy();
    //outFile.close();
    engine->destroy();
    builder->destroy();
    pluginFactory.destroyPlugin();
    shutdownProtobufLibrary();
}

void TensorNet::savePlan(const std::string& deployFile,
                         const std::string& modelFile,
                         const std::vector<std::string>& outputs,
                         unsigned int maxBatchSize){
    cout <<"saving plan file to "<<SERIALIZE_FILE<<endl;
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();
    ICaffeParser* parser = createCaffeParser();
    parser->setPluginFactory(&pluginFactory);
    bool fp16 = builder->platformHasFastFp16();
    if (fp16) {
        printf("Using Fp16 mode\n");
    }
    const IBlobNameToTensor *blobNameToTensor = parser->parse(deployFile.c_str(),
            modelFile.c_str(), *network, fp16?nvinfer1::DataType::kHALF :  nvinfer1::DataType::kFLOAT);
        assert(blobNameToTensor != nullptr);
    for (auto& s : outputs) network->markOutput(*blobNameToTensor->find(s.c_str()));

    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(4 << 30);
    builder->setHalf2Mode(fp16);


    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    network->destroy();
    parser->destroy();

    cout <<"Serializing to "<<SERIALIZE_FILE<<endl;
    gieModelStream = engine->serialize();
    std::ofstream outFile;
    outFile.open(SERIALIZE_FILE);
    std::stringstream  outstream;
    outstream.write((const char*)gieModelStream->data(), gieModelStream->size());
    outFile << outstream.rdbuf();
    gieModelStream->destroy();
    outFile.close();
    engine->destroy();
    builder->destroy();
    pluginFactory.destroyPlugin();
    shutdownProtobufLibrary();
}

void TensorNet::loadPlan(const std::string& planFile) {
    cout <<"Loading plan file "<<planFile<<endl;
    infer = createInferRuntime(gLogger);
    std::ifstream ifile(planFile);
    assert(ifile);
    std::stringstream serial_data;
    serial_data<<ifile.rdbuf();
    ifile.close();
    serial_data.seekg(0, std::ios::end);
    const int model_size = serial_data.tellg();
    serial_data.seekg(0, std::ios::beg);
    void* modelMem = malloc(model_size);
    assert(modelMem != nullptr);
    serial_data.read((char*)modelMem, model_size);
    engine = infer->deserializeCudaEngine(modelMem, model_size, &pluginFactory);
    free(modelMem);
    printf("Bindings after deserializing:\n"); 
    for (int bi = 0; bi < engine->getNbBindings(); bi++) {
        if (engine->bindingIsInput(bi) == true) printf("Binding %d (%s): Input.\n",  bi, engine->getBindingName(bi));
        else printf("Binding %d (%s): Output.\n", bi, engine->getBindingName(bi));
    }
}

void TensorNet::createInference()
{
    infer = createInferRuntime(gLogger);
    engine = infer->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), &pluginFactory);

    printf("Bindings after deserializing:\n"); 
    for (int bi = 0; bi < engine->getNbBindings(); bi++) {
        if (engine->bindingIsInput(bi) == true) printf("Binding %d (%s): Input.\n",  bi, engine->getBindingName(bi));
        else printf("Binding %d (%s): Output.\n", bi, engine->getBindingName(bi));
    }
}

void TensorNet::imageInference(void** buffers, int nbBuffer, int batchSize, float* result)
{
    assert(engine->getNbBindings()==nbBuffer);

    IExecutionContext* context = engine->createExecutionContext();
    //context->setProfiler(&gProfiler);
    std::cout <<"execute .."<<std::endl;
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    //context->execute(batchSize, buffers);
    context->enqueue(batchSize, buffers, stream, nullptr);
    DimsCHW dimsout = getTensorDims(REGION_OUTPUT);
    const int output_size = batchSize*dimsout.c()*dimsout.h()*dimsout.w();
    CHECK(cudaMemcpyAsync(result, buffers[engine->getBindingIndex(REGION_OUTPUT)],output_size*sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    context->destroy();
}


void TensorNet::timeInference(int iteration, int batchSize)
{
    int inputIdx = 0;
    size_t inputSize = 0;

    void* buffers[engine->getNbBindings()];

    for (int b = 0; b < engine->getNbBindings(); b++) {
        DimsCHW dims = static_cast<DimsCHW&&>(engine->getBindingDimensions(b));
        size_t size = batchSize * dims.c() * dims.h() * dims.w() * sizeof(float);
        CHECK(cudaMallocManaged(&buffers[b], size));

        if(engine->bindingIsInput(b) == true)
        {
            inputIdx = b;
            inputSize = size;
        }
    }

    IExecutionContext* context = engine->createExecutionContext();
    context->setProfiler(&gProfiler);

    CHECK(cudaMemset(buffers[inputIdx], 0., inputSize));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));


    //for (int i = 0; i < iteration;i++) context->enqueue(batchSize, buffers);
    for (int i = 0; i < iteration;i++) context->enqueue(batchSize, buffers, stream, nullptr);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    context->destroy();
    for (int b = 0; b < engine->getNbBindings(); b++) CHECK(cudaFree(buffers[b]));
}

DimsCHW TensorNet::getTensorDims(const char* name)
{
    for (int b = 0; b < engine->getNbBindings(); b++) {
        if( !strcmp(name, engine->getBindingName(b)) )
            return static_cast<DimsCHW&&>(engine->getBindingDimensions(b));
    }
    return DimsCHW{0,0,0};
}

void TensorNet::printTimes(int iteration)
{
    gProfiler.printLayerTimes(iteration);
}

void TensorNet::destroy()
{
    pluginFactory.destroyPlugin();
    engine->destroy();
    infer->destroy();
}






float* allocateMemory(DimsCHW dims, char* info, int batchSize)
{
    float* ptr;
    size_t size;
    size = batchSize * dims.c() * dims.h() * dims.w();
    std::cout << "Allocate memory: " << info <<" size "<<size<< std::endl;
    assert(!cudaMallocManaged(&ptr, size*sizeof(float)));
    return ptr;
}

void printDims(DimsCHW dims, const char* str){
    printf("%s\n", str);
    printf("dims c(%d) h(%d) w(%d)\n", dims.c(), dims.h(), dims.w());
}
//********************
//main
//*******************
int main(int argc, char **argv) {
    std::cout << "Tensorrt start for "<< model<<std::endl;
    TensorNet tensornet;
    //tensornet.caffeToTRTModel(model, weight,std::vector<std::string>{REGION_OUTPUT}, BATCH_SIZE);
    //tensornet.createInference();
    tensornet.savePlan(model, weight,std::vector<std::string>{REGION_OUTPUT}, MAX_BATCH_SIZE);
    tensornet.loadPlan(SERIALIZE_FILE);

    DimsCHW dimsData = tensornet.getTensorDims(INPUT_BLOB_NAME);
    DimsCHW dimsout = tensornet.getTensorDims(REGION_OUTPUT);
    printDims(dimsData, "Input blob");
    printDims(dimsout, "REGION_OUTPUT");
    const int input_size = dimsData.c() * dimsData.w() * dimsData.h();
    float *img = (float*)malloc(sizeof(float)*input_size*BATCH_SIZE);
    float *result = (float*)malloc(sizeof(float)*dimsout.c()*dimsout.h()*dimsout.w()*BATCH_SIZE);

    Mat img_u8, img_f32;
    img_u8 = imread(pic_path, CV_LOAD_IMAGE_COLOR);
    img_u8.convertTo(img_f32, CV_32F);
    int datum_index = 0;
    for (int b = 0; b < BATCH_SIZE; b++ ) {
        for (int h = 0; h < img_f32.rows; ++h) {
            const float *ptr = img_f32.ptr<float>(h);
            int img_index = 0;
            for (int w = 0; w < img_f32.cols; ++w) {
                for (int c = 0; c < img_f32.channels(); ++c) {
                    //int datum_index = (c*img_f32.rows + h) * img_f32.cols + w;
                    img[datum_index] = static_cast<float>(ptr[img_index++]);
                    datum_index++;
                }
            }
        }
    }


    printf("\n");
    assert((dimsData.c()*dimsData.h()*dimsData.w() == (img_f32.cols*img_f32.rows*img_f32.channels())));
    float *input = allocateMemory(dimsData, (char*)"input blob", BATCH_SIZE);
    float *output = allocateMemory(dimsout, (char*)"region_output", BATCH_SIZE);
    cudaMemcpy(input, img ,BATCH_SIZE*dimsData.c()*dimsData.h()*dimsData.w()*sizeof(float), cudaMemcpyHostToDevice);
    printf("img data\n");
    //for (int i = 0; i < 10; i++) {
        //printf("%f %f %f   ", img[i], img[i+input_size], img[i+2*input_size]);
    //}
    //printf("\n");
/*    for (int i = 0; i < 10; i++) {*/
        //printf("%f ", img[i]);
    //}
    /*printf("\n");*/
        
    void* buffers[] = {input, output};
    std::cout << "start inference .."<<std::endl;
    tensornet.imageInference(buffers, 2, BATCH_SIZE, result);
    printf("output  ");
    for (int i = 0; i < BATCH_SIZE*dimsout.c()*dimsout.h()*dimsout.w(); i++) {
        printf("%f ", result[i]);
    }
    printf("\n");
    std::cout <<"time inference Done.."<<endl;
        
    std::cout <<"start profile..."<<endl;
    struct timeval start, end;
    gettimeofday(&start, NULL);
#ifndef DEBUG
    tensornet.timeInference(TIMING_ITERATIONS, BATCH_SIZE);
#endif
    gettimeofday(&end, NULL);
    float time_use = 1000.0*(end.tv_sec - start.tv_sec) + (float)(end.tv_usec - start.tv_usec)/1000.0;
    cout <<"Inference Done in "<<time_use / TIMING_ITERATIONS<<"ms"<<endl;
    tensornet.printTimes(TIMING_ITERATIONS);
        
    //destroy
/*    tensornet.destroy();*/
    //free(result);
    //std::cout <<"Done."<<std::endl;
    //printf("test Detector\n");
    //DetectNet *net = new DetectNet();
    //net->loadNet(SERIALIZE_FILE);
    //net->inference(img, 1, result);
    //for (int i = 0; i < BATCH_SIZE*dimsout.c()*dimsout.h()*dimsout.w(); i++) {
        //printf("%f ", result[i]);
    /*}*/
    return 0;
}
