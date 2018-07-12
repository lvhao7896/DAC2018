#include "plugin.hpp"
//#include "test.hpp"

//static const float biases[10] = {1.08, 1.19,  3.42, 4.41,  6.63, 11.38,  9.42,5.11,  16.62, 10.52};
//static const float biases[10] = {0.52573, 0.677385,  1.87446, 2.06253, 3.33843,5.47434, 7.88282,3.52778, 9.77052, 9.16828};
//static const float biases[10] = {0.34,0.56, 0.63,1.07, 1.04,2.46, 1.27,1.13, 2.78,2.19};
//static const float biases[2*BOX_NUM] = {0.28,0.42, 0.39,1.14, 0.66,0.74, 0.76,1.76, 1.06,1.07, 1.09,2.69, 1.56,1.58, 1.73,4.45, 2.61,2.33, 4.83,4.11};
static const float biases[2*BOX_NUM] = {0.21,0.29, 0.27,0.73, 0.50,0.50, 0.52,1.12, 0.74,1.70, 0.85,0.78, 1.10,2.82, 1.30,1.21, 2.11,1.63, 3.32,2.69};

nvinfer1::IPlugin* PluginFactory::createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights){
    assert(isPlugin(layerName));
    LayerType layerType = getLayerType(layerName);
    nvinfer1::IPlugin* tmp = nullptr;
    switch (layerType) {
        case LEAKYRELULAYER:
            assert(pLeakyReluLayer.get() == nullptr);
            pLeakyReluLayer = std::unique_ptr<LeakyReLULayer> (new LeakyReLULayer());
            tmp = pLeakyReluLayer.get();
            leakyreluvec.push_back(std::move(pLeakyReluLayer));
            return tmp;
            break;
        case PREPROCESSLAYER:
            assert(preprocessLayer.get() == nullptr);
            preprocessLayer = std::unique_ptr<PreprocessLayer>(new PreprocessLayer());
            return preprocessLayer.get();
            break;
        case POSTPROCESSLAYER:
            assert(postProcessLayer.get() == nullptr);
            postProcessLayer = std::unique_ptr<PostProcessLayer>(new PostProcessLayer());
            return postProcessLayer.get();
            break;
        case REORGLAYER:
            assert(reorgLayer.get() == nullptr);
            reorgLayer = std::unique_ptr<ReorgLayer>(new ReorgLayer());
            return reorgLayer.get();
            break;
        case DEPTHWISELAYER_S1:
            cout <<"layer stride 1: "<<layerName<<" weight num "<<nbWeights<<" count: "<<weights[0].count<<endl;
            assert(std::find_if(depLayers_s1.begin(), depLayers_s1.end(), [&](const DepthwiseLayer_s1& r) {return r.first == layerName;}) == depLayers_s1.end());
            dLayer_s1 = std::unique_ptr<DepthwiseConvLayer_s1>(new DepthwiseConvLayer_s1(weights, nbWeights));
            tmp  = dLayer_s1.get();
            depLayers_s1.push_back(std::make_pair(layerName, std::move(dLayer_s1))); 
            return tmp;
        case DEPTHWISELAYER_S2:
            cout <<"layer stride 2: "<<layerName<<" weight num "<<nbWeights<<" count :"<<weights[0].count<<endl;
            assert(std::find_if(depLayers_s2.begin(), depLayers_s2.end(), [&](const DepthwiseLayer_s2& r) {return r.first == layerName;}) == depLayers_s2.end());
            dLayer_s2 = std::unique_ptr<DepthwiseConvLayer_s2>(new DepthwiseConvLayer_s2(weights, nbWeights));
            tmp = dLayer_s2.get();
            depLayers_s2.push_back(std::make_pair(layerName, std::move(dLayer_s2))); 
            return tmp;
        case OUTPUTLAYER:
            assert(outputLayer.get() == nullptr);
            outputLayer = std::unique_ptr<OutputLayer>(new OutputLayer());
            return outputLayer.get();
#ifdef DEBUG
        case TESTLAYER:
            assert(testLayer.get() == nullptr);
            testLayer = std::unique_ptr<TestLayer>(new TestLayer(weights, nbWeights));
            return testLayer.get();
#endif
        default:
            printf("Not supportted layer\n");
            assert(0);
    }
}

IPlugin* PluginFactory::createPlugin(const char* layerName, const void *serialData, size_t serialLength) {
    assert(isPlugin(layerName));
    LayerType layerType = getLayerType(layerName);
    IPlugin * tmp = nullptr;
    switch (layerType) {
        case LEAKYRELULAYER:
            assert(pLeakyReluLayer.get() == nullptr);
            pLeakyReluLayer = std::unique_ptr<LeakyReLULayer> (new LeakyReLULayer(serialData, serialLength));
            tmp = pLeakyReluLayer.get();
            leakyreluvec.push_back(std::move(pLeakyReluLayer));
            return tmp;
            break;
        case PREPROCESSLAYER:
            assert(preprocessLayer.get() == nullptr);
            preprocessLayer = std::unique_ptr<PreprocessLayer>(new PreprocessLayer());
            return preprocessLayer.get();
            break;
        case POSTPROCESSLAYER:
            assert(postProcessLayer.get() == nullptr);
            postProcessLayer = std::unique_ptr<PostProcessLayer>(new PostProcessLayer());
            return postProcessLayer.get();
            break;
        case REORGLAYER:
            assert(reorgLayer.get() == nullptr);
            reorgLayer = std::unique_ptr<ReorgLayer>(new ReorgLayer());
            return reorgLayer.get();
            break;
        case DEPTHWISELAYER_S1:
            assert(std::find_if(depLayers_s1.begin(), depLayers_s1.end(), [&](const DepthwiseLayer_s1& r) {return r.first == layerName;}) == depLayers_s1.end());
            dLayer_s1 = std::unique_ptr<DepthwiseConvLayer_s1>(new DepthwiseConvLayer_s1(serialData, serialLength));
            tmp = dLayer_s1.get();
            depLayers_s1.push_back(std::make_pair(layerName, std::move(dLayer_s1))); 
            return tmp;
            break;
        case DEPTHWISELAYER_S2:
            assert(std::find_if(depLayers_s2.begin(), depLayers_s2.end(), [&](const DepthwiseLayer_s2& r) {return r.first == layerName;}) == depLayers_s2.end());
            dLayer_s2 = std::unique_ptr<DepthwiseConvLayer_s2>(new DepthwiseConvLayer_s2(serialData, serialLength));
            tmp = dLayer_s2.get();
            depLayers_s2.push_back(std::make_pair(layerName, std::move(dLayer_s2))); 
            return tmp;
            break;
        case OUTPUTLAYER:
            assert(outputLayer.get() == nullptr);
            outputLayer = std::unique_ptr<OutputLayer>(new OutputLayer());
            return outputLayer.get();
#ifdef DEBUG
        case TESTLAYER:
            assert(testLayer.get() == nullptr);
            testLayer = std::unique_ptr<TestLayer>(new TestLayer(serialData, serialLength));
            return testLayer.get();
#endif
        default:
            printf("Not supportted layer\n");
            assert(0);
    }
}
 

bool PluginFactory::isPlugin(const char *name){
    return (!strcmp(name, "preprocess"))
        || (!strcmp(name, "postprocess"))
        || (!strcmp(name, "reorg"))
        || (!strcmp(name, "region_output"))
        || (find_if(dep_conv_s1_layers_name.begin(), dep_conv_s1_layers_name.end(), [&](const string & s){return s == string(name);}) != dep_conv_s1_layers_name.end())
        || (find_if(dep_conv_s2_layers_name.begin(), dep_conv_s2_layers_name.end(), [&](const string & s){return s == string(name);}) != dep_conv_s2_layers_name.end())
        || (strstr(name, "leaky"))
#ifdef DEBUG 
        || (find_if(test_layers_name.begin(), test_layers_name.end(), [&](const string&s){return s == string(name);}) != test_layers_name.end())
#endif
        ;
}

PluginFactory::LayerType PluginFactory::getLayerType(const char* name){
    if (strstr(name, "leaky")) {
        return LEAKYRELULAYER;
    }
    if(!strcmp(name, "preprocess")){
        return PREPROCESSLAYER;
    }
    if(!strcmp(name, "postprocess")){
        return POSTPROCESSLAYER;
    }
    if(!strcmp(name, "reorg")){
        return REORGLAYER;
    }
    if(find_if(dep_conv_s1_layers_name.begin(), dep_conv_s1_layers_name.end(), [&](const string& s) {return s == string(name);}) != dep_conv_s1_layers_name.end()) {
        return DEPTHWISELAYER_S1;
    }
    if(find_if(dep_conv_s2_layers_name.begin(), dep_conv_s2_layers_name.end(), [&](const string& s) {return s == string(name);}) != dep_conv_s2_layers_name.end()) {
        return DEPTHWISELAYER_S2;
    }
    if(!strcmp(name, "region_output")){
        return OUTPUTLAYER;
    }
#ifdef DEBUG
    if(find_if(test_layers_name.begin(), test_layers_name.end(), [&](const string&s){return s == string(name);}) != test_layers_name.end()){
        return TESTLAYER;
    }
#endif
    return NOT_SUPPORTED;
}

void PluginFactory::destroyPlugin(){
    preprocessLayer.release();
    preprocessLayer = nullptr;
    postProcessLayer.release();
    postProcessLayer = nullptr;
    reorgLayer.release();
    reorgLayer = nullptr;
    outputLayer.release();
    outputLayer = nullptr;
    for (auto it = depLayers_s1.begin(); it != depLayers_s1.end(); it++) {
        (*it).second.release();
        (*it).second = nullptr;
    }
    depLayers_s1.clear();
    for (auto it = depLayers_s2.begin(); it != depLayers_s2.end(); it++) {
        (*it).second.release();
        (*it).second = nullptr;
    }
    depLayers_s2.clear();
    for (auto it = leakyreluvec.begin(); it != leakyreluvec.end(); it++) {
        (*it).release();
        (*it) = nullptr;
    }
    leakyreluvec.clear();
#ifdef DEBUG
    if(testLayer != nullptr){
        testLayer.release();
        testLayer = nullptr;
    }
#endif
}


/***************************/
//preprocessLayer
/***************************/
void preprocess(int count, float *input, float* output);

Dims PreprocessLayer::getOutputDimensions(int index, const Dims* inputs, int nbInputDims){
    assert(nbInputDims == 1);
    return DimsCHW(3,RESIZE_H, RESIZE_W);
}

int PreprocessLayer::initialize(){
    return 0;
}

void PreprocessLayer::configure(const Dims*inputs, int nbInputs, const Dims*outputs, int nbOutputs, int maxBatch) {
#ifdef DEBUG
    std::cout<<"PreprocessLayer  configure"<<std::endl;
#endif
}

size_t PreprocessLayer::getSerializationSize(){
    return 0;
}

void PreprocessLayer::serialize(void* buff){
    return;
}

int PreprocessLayer::enqueue(int batchSize, const void* const* inputs, void **outputs, void *, cudaStream_t stream){
    //CHECK(cudaThreadSynchronize());
#ifdef DEBUG
    cout<<"in preprocess enqueue"<<endl;
#endif
    preprocess(batchSize*RESIZE_H*RESIZE_W*CHANNEL, (float*)inputs[0], (float*)outputs[0]);
#ifdef DEBUG_PREPROCESS
    const int count = RESIZE_H*RESIZE_W*CHANNEL;
    float * tmp = (float*)malloc(sizeof(float)*count*batchSize);
    CHECK(cudaMemcpy(tmp, (float*)outputs[0], sizeof(float) * count*batchSize, cudaMemcpyDeviceToHost));
    for (int j = 0; j < batchSize; j++) {
        for (int i = 0; i < 360*3; i++) {
            printf("%f ", tmp[i+j*count]);
        }
        printf("\n");
    }
    free(tmp);
#endif
    return 0;
}

/***************************/
//reorgLayer
/***************************/
void reorg(float *input, float* ouput,int batchSize);
Dims ReorgLayer::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) {
    assert(nbInputDims == 1);
    return DimsCHW(inputs->d[0]*4, inputs->d[1]/2, inputs->d[2]/2);
}

int ReorgLayer::enqueue(int batchSize, const void* const* inputs, void **output, void *, cudaStream_t stream) {
    //CHECK(cudaThreadSynchronize());
#ifdef DEBUG
    cout <<"in reorg layer enqueue"<<endl;
#endif
    reorg((float*)inputs[0], (float*)output[0], batchSize);
    return 0;
}

void ReorgLayer::configure(const Dims*inputs, int nbInputs, const Dims*outputs, int nbOutputs, int maxBatch) {
#ifdef DEBUG
    std::cout<<"ReorgLayer configure"<<std::endl;
#endif
}

/***************************/
//postprocessLayer
/***************************/
void postprocess(float* input, float* output, int batch);
Dims PostProcessLayer::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) {
    assert(nbInputDims == 1);
    return DimsCHW(inputs->d[1], inputs->d[2], inputs->d[0]);
}

int PostProcessLayer::enqueue(int batchSize, const void *const *inputs, void **output, void*, cudaStream_t stream) {
    //CHECK(cudaThreadSynchronize());
#ifdef DEBUG
    cout <<"in post process enqueue"<<endl;
#endif
    postprocess((float*)inputs[0], (float*)output[0], batchSize);
#ifdef DEBUG_POSTPROCESS_INPUT
    printf("postprocess input\n");
    const int count = RESIZE_H/32*RESIZE_W/32*BOX_NUM*(5+CLASS_NUM);
    float *tmp = (float*)malloc(sizeof(float) * count);
    cudaMemcpy(tmp, (float*)inputs[0], sizeof(float) * count, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 5*14; i++) {
        printf("%f ", tmp[i]);
    }
    printf("\n");
    free(tmp);
#endif
#ifdef DEBUG_POSTPROCESS
    printf("postprocess output\n");
    const int count = RESIZE_H/32*RESIZE_W/32*BOX_NUM*(5+CLASS_NUM);
    float * tmp = (float*)malloc(sizeof(float)*count*batchSize);
    cudaMemcpy(tmp, (float*)output[0], sizeof(float) * count*batchSize, cudaMemcpyDeviceToHost);
    for (int j = 0; j < batchSize; j++) {
        for (int i = 0; i < 5*14; i++) {
            printf("%f ", tmp[i+count*j]);
        }
        printf("\n");
    }
    printf("\n");
    free(tmp);
#endif
    return 0;
}

void PostProcessLayer::configure(const Dims*inputs, int nbInputs, const Dims*outputs, int nbOutputs, int maxBatch) {
#ifdef DEBUG
    std::cout<<"PostProcessLayer configure"<<std::endl;
#endif
}

/***************************/
//leakyreluLayer
/***************************/
void leakyrelu(float*input, float* output, int count);
Dims LeakyReLULayer::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) {
    assert(nbInputDims == 1);
    return DimsCHW(inputs->d[0], inputs->d[1], inputs->d[2]);
}

int LeakyReLULayer::enqueue(int batchSize, const void *const *inputs, void **output, void *, cudaStream_t stream) {
    //CHECK(cudaThreadSynchronize());
#ifdef DEBUG
    cout <<"in leaky relu enqueue"<<endl;
#endif
    leakyrelu((float*)inputs[0], (float*)output[0],batchSize*count);
#ifdef DEBUG_LEAKYRELU
    float * tmp = (float*)malloc(sizeof(float)*count*batchSize);
    cudaMemcpy(tmp, (float*)output[0], sizeof(float) * count*batchSize, cudaMemcpyDeviceToHost);
    for (int j = 0; j < batchSize; j++) {
        for (int i = 0; i < 20; i++) {
            printf("%f ", tmp[i+count*j]);
        }
        printf("\n");
    }
    printf("\n");
    free(tmp);
#endif
    return 0;
}

void LeakyReLULayer::configure(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) {
    assert(inputs->nbDims == 3);
    count = nbInputs * inputs->d[0] * inputs->d[1] * inputs->d[2];
#ifdef DEBUG
    printf("leakyrelu NCHW:(%d %d %d %d) \tcount :%d\n",nbInputs, inputs->d[0], inputs->d[1], inputs->d[2],  count);
#endif
}

size_t LeakyReLULayer::getSerializationSize() {
    return sizeof(int) * 1;
}

void LeakyReLULayer::serialize(void *buff) {
    char *d = reinterpret_cast<char*>(buff), *a = d;
    write(d, count);
    assert(d == a + getSerializationSize());
}

LeakyReLULayer::LeakyReLULayer(const void *data, size_t length) {
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    count = read<int>(d);
    assert(d == a +length);
}

/***************************/
//depthwiseConvLayer
/***************************/
void depthwiseConv(int batch, int c, int w, int h, int stride, float* input, float* weight, float* output);

DepthwiseConvLayer_s1::DepthwiseConvLayer_s1(const nvinfer1::Weights *weights, int nbWeights) {
    //for simpcity just assume no bias
    assert(nbWeights == 1);
    assert(weights != nullptr);
    assert(weights[0].count != 0);
    kernelWeight = copyToDevice(weights[0].values, weights[0].count);
}

DepthwiseConvLayer_s1::DepthwiseConvLayer_s1(const void *data, size_t length) {
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    channels = read<int>(d);
    height = read<int>(d);
    width = read<int>(d);
    const int kernel_h = KERNEL_SIZE;
    const int kernel_w = KERNEL_SIZE;
    kernelWeight = deserializeToDevice(d, kernel_h*kernel_w*channels);
    assert(d == a + length);
}

Dims DepthwiseConvLayer_s1::getOutputDimensions(int idnex, const Dims* inputs, int nbInputsDims) {
    assert(nbInputsDims == 1);
    return DimsCHW(inputs->d[0], inputs->d[1], inputs->d[2]);
}

void DepthwiseConvLayer_s1::configure(const Dims*inputs, int nbInputs, const Dims*outputs, int nbOutputs, int maxBatch) {
    channels = inputs->d[0];
    height = inputs->d[1];
    width = inputs->d[2];
}

size_t DepthwiseConvLayer_s1::getSerializationSize() {
    return sizeof(int) * 3 + kernelWeight.count*sizeof(float);
}

void DepthwiseConvLayer_s1::serialize(void *buff) {
    char *d = reinterpret_cast<char*>(buff), *a = d;
    write(d, channels);
    write(d, height);
    write(d, width);
    serializeFromDevice(d, kernelWeight);
    assert(d == a + getSerializationSize());
}

 
int DepthwiseConvLayer_s1::enqueue(int batchSize, const void* const *inputs, void **outputs, void *workspace, cudaStream_t stream) {
    //CHECK(cudaThreadSynchronize());
#ifdef DEBUG_DEPTHWISE
    printf("in Depthwise_s1 enqueue\n");
    printf("channel(%d) height(%d) width(%d) stride(%d)\n", channels, height, width, stride);
#endif
    depthwiseConv(batchSize, channels, width, height, stride, (float*)inputs[0], (float*)kernelWeight.values, (float*)outputs[0]);
#ifdef DEBUG_DEPTHWISE
    int count = channels * height * width;
    printf("\ndepthwiseconv input \n");
    float * tmp = (float*)malloc(sizeof(float)*count);
    cudaMemcpy(tmp, (float*)outputs[0], sizeof(float) * count, cudaMemcpyDeviceToHost);
/*    for (int i = 0; i < channels; i++) {*/
        //for(int j = 0; j < height; j++) {
            //for (int k = 0; k < width; k++) {
                //printf("%f ", tmp[i]);
            //}
            //printf("\n");
        //}
        //printf("\n");
    /*}*/
    for (int i = 0; i < 30 ; i ++) {
        printf("%.10f ", tmp[i]);
    }
    printf("\n");
    free(tmp);
#endif
    return 0;

}


//stride 2 implementation

DepthwiseConvLayer_s2::DepthwiseConvLayer_s2(const nvinfer1::Weights *weights, int nbWeights) {
    //for simpcity just assume no bias
    assert(nbWeights == 1);
    assert(weights != nullptr);
    assert(weights[0].count != 0);
    kernelWeight = copyToDevice(weights[0].values, weights[0].count);
}

DepthwiseConvLayer_s2::DepthwiseConvLayer_s2(const void *data, size_t length) {
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    channels = read<int>(d);
    height = read<int>(d);
    width = read<int>(d);
    const int kernel_h = KERNEL_SIZE;
    const int kernel_w = KERNEL_SIZE;
    kernelWeight = deserializeToDevice(d, kernel_h*kernel_w*channels);
    assert(d == a + length);
}

Dims DepthwiseConvLayer_s2::getOutputDimensions(int idnex, const Dims* inputs, int nbInputsDims) {
    assert(nbInputsDims == 1);
    assert((inputs->d[1] % 2 == 0) && (inputs->d[2] % 2 == 0));
    return DimsCHW(inputs->d[0], inputs->d[1]/2, inputs->d[2]/2);
}

void DepthwiseConvLayer_s2::configure(const Dims*inputs, int nbInputs, const Dims*outputs, int nbOutputs, int maxBatch) {
    channels = inputs->d[0];
    height = inputs->d[1];
    width = inputs->d[2];
}

size_t DepthwiseConvLayer_s2::getSerializationSize() {
    return sizeof(int) * 3 + kernelWeight.count*sizeof(float);
}

void DepthwiseConvLayer_s2::serialize(void *buff) {
    char *d = reinterpret_cast<char*>(buff), *a = d;
    write(d, channels);
    write(d, height);
    write(d, width);
    serializeFromDevice(d, kernelWeight);
    assert(d == a + getSerializationSize());
}

 
int DepthwiseConvLayer_s2::enqueue(int batchSize, const void* const *inputs, void **outputs, void *workspace, cudaStream_t stream) {
    //CHECK(cudaThreadSynchronize());
#ifdef DEBUG_DEPTHWISE
    printf("in Depthwise_s2 enqueue\n");
    printf("channel(%d) height(%d) width(%d) stride(%d)\n", channels, height, width, stride);
#endif
    depthwiseConv(batchSize, channels, width, height, stride, (float*)inputs[0], (float*)kernelWeight.values, (float*)outputs[0]);
#ifdef DEBUG_DEPTHWISE
    int count = channels * height * width;
    printf("\ndepthwiseconv input \n");
    float * tmp = (float*)malloc(sizeof(float)*count);
    cudaMemcpy(tmp, (float*)inputs[0], sizeof(float) * count, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 30; i++) {
        printf("%.10f ", tmp[i]);
    }
    printf("\n");
    free(tmp);
#endif
    return 0;
}



/***************************/
//OutputLayer
/***************************/
void OutputLayer::configure(const Dims*inputs, int nbInputs, const Dims*outputs, int nbOutputs, int maxBatch) {
#ifdef DEBUG
    std::cout<<"OutputLayer configure"<<std::endl;
#endif
}

Dims OutputLayer::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) {
    assert(nbInputDims == 1);
    return DimsCHW(1, 1, 4);
}

struct Comperator {
    explicit Comperator(const vector<sbbox> &ref) {
        ptr = &ref;
    }
    const vector<sbbox> *ptr;
    bool operator()(const int lhs, const int rhs) const {
        return (*ptr)[lhs].score > (*ptr)[rhs].score;
    }
};

inline float overlap(float x1, float w1, float x2, float w2) {
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = std::max(l1, l2);
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = std::min(r1, r2);
    return right - left;
}

inline float box_inter(const sbbox &b1, const sbbox &b2) {
    float w = overlap(b1.x, b1.w, b2.x, b2.w);
    float h = overlap(b1.y, b1.h, b2.y, b2.h);
    if (w < 0 || h < 0) return 0;
    return w*h;
}

inline float getIOU(const sbbox &b1, const sbbox &b2) {
    float Sa = b1.w * b1.h;
    float Sb = b2.w * b2.h;
    float interS = box_inter(b1, b2);
    return interS/ (Sa + Sb - interS);
}

vector<int> nms_cpu(const std::vector<sbbox> &bbs, float overlapTh, bool has_sorted) {
    int num = bbs.size();
    if (num == 0) return vector<int>();
    vector<bool> jump(num, false);
    vector<int> idx(num);
    for (int i = 0; i < num; i++) {
        idx[i] = i;
    }
    if (!has_sorted) {
        std::sort(idx.begin(), idx.end(), Comperator(bbs));
    }
    float overlap;
    int i = 0;
    while(true) {
        for (; i < num && jump[i]; i++) {;}
        if (i >= num) break;
        const sbbox &bi = bbs[idx[i]];
        for (int j = num - 1; j > i; j--) {
            if (jump[j])
                continue;
            const sbbox &bj = bbs[idx[j]];
            overlap = getIOU(bi, bj);
            if(overlap > overlapTh) {
                jump[j] = true;
            }
        }
        i++;
    }
        
    vector<int> keep;
    for (i = 0; i < num; i++) {
        if (!jump[i]) {
            keep.push_back(idx[i]);
        }
    }
    return keep;
}

sbbox getRegion_box(const float *x, const float* biases, const int cx, const int cy, const int height, const int width) {
    float bx = cx + x[0];
    float by = cy + x[1];
    float bw = biases[0] * x[2];
    float bh = biases[1] * x[3];
    return sbbox(bx/ width, by/height, bw/width, bh/height, 0);
}


vector<sbbox> getPredBoxes(const int height, const int width, const int boxes_of_each_grid,
                        const int classes, const float* bottom_data, const float*biases,
                        float thresh_prob, float thres_nms) {
    vector<sbbox> total_boxes;
    vector<sbbox> result_boxes;
    float probs[height * width * boxes_of_each_grid][classes];
    for (int i = 0; i < height*width; i++) {
        int row = i / width;
        int col = i % width;
        for (int n = 0; n <boxes_of_each_grid; n++ ) {
            int index = i * boxes_of_each_grid + n;
            int box_index = index * (classes + 5);
            sbbox sbbox_tmp = getRegion_box(bottom_data + box_index, biases + 2*n, col, row, height, width);
            total_boxes.push_back(sbbox_tmp);
            float scale = bottom_data[box_index + 4];
            int class_index = box_index + 5;
            for (int j = 0; j < classes; j++) {
                float prob = scale * bottom_data[class_index + j];
                probs[index][j] = prob > thresh_prob ? prob : 0;
            }
        }
    }
        
    for (int i = 0; i < classes; i++) {
        vector<sbbox> nms_boxes;
        for (int j = 0; j < total_boxes.size(); j++) {
            if (probs[j][i] > thresh_prob) {
                total_boxes[j].score = probs[j][i];
                nms_boxes.push_back(total_boxes[j]);
            }
        }
        vector <int> ret = nms_cpu(nms_boxes, thres_nms, false);
        for (int k = 0; k < ret.size(); k++) {
            result_boxes.push_back(nms_boxes[ret[k]]);
        }
    }
    return result_boxes;
}



int OutputLayer::enqueue(int batchSize, const void *const *inputs, void ** output, void *, cudaStream_t stream) {
    //CHECK(cudaThreadSynchronize());
#ifdef DEBUG
    printf("in outputlayer enqueu\n");
#endif
    const float *bottom_data = (float*)inputs[0];
    float *top_data = (float*)output[0];
    const int height = RESIZE_H / 32;
    const int width = RESIZE_W / 32;
    const int channel = BOX_NUM*(5 + CLASS_NUM);
    const int classes = CLASS_NUM;
    const int batch_num = batchSize;
    const float thres_nms = THRESH_NMS;
    const float thres_prob = THRESH_PROB;
    vector<vector<sbbox> > result_boxes_with_batch;
    const int boxes_of_each_grid = BOX_NUM;
    const int feature_map_szie = height * width * channel;
    float data_tmp[feature_map_szie * batch_num];
    CHECK(cudaMemcpy(data_tmp, bottom_data, feature_map_szie * batch_num*sizeof(float), cudaMemcpyDeviceToHost));
#ifdef DEBUG_OUTPUTLAYER
    printf("feature_map_szie : %d\n", feature_map_szie);
    printf("output layer input\n");
    for (int j = 0; j < 13*7; j++ ){
    for (int i = 0; i < 70; i++) {
        printf("%f ", data_tmp[i+j*70]);
    }
        printf("\n");
    }
    printf("\n");
#endif
    for (int batch = 0; batch < batch_num; batch++) {
        vector<sbbox>result_boxes = getPredBoxes(height, width, boxes_of_each_grid, 
                                classes, data_tmp + feature_map_szie * batch,
                                biases, thres_prob, thres_nms);
        //printf("box size %d\n", (int)result_boxes.size());
        result_boxes_with_batch.push_back(result_boxes);
    }
    int count = 0;
    int box_size = 4;
    for (int i = 0; i < result_boxes_with_batch.size(); i++) {
        vector<sbbox> result_boxes = result_boxes_with_batch[i];
            top_data[box_size * count + 0] = 0.5;
            top_data[box_size * count + 1] = 0.5;
            top_data[box_size * count + 2] = 0.3;
            top_data[box_size * count + 3] = 0.3;
            //top_data[box_size * count + 4] = 0;
            //top_data[box_size * count + 5] = 0;
            //top_data[box_size * count + 6] = 0;
            for (int j = 0; j < result_boxes.size() && j < 1; j++) {
            //for (int j = 0; j < result_boxes.size(); j++) {
                //printf("in result_boxes\n");
                //printf("idx %d (%f %f %f %f %f)\n", result_boxes[j].x, result_boxes[j].y,
                        //result_boxes[j].w, result_boxes[j].h, result_boxes[j].score);
                top_data[box_size * count + 0] = result_boxes[j].x;
                top_data[box_size * count + 1] = result_boxes[j].y;
                top_data[box_size * count + 2] = result_boxes[j].w;
                top_data[box_size * count + 3] = result_boxes[j].h;
                //top_data[box_size * count + 4] = result_boxes[j].score;
                //top_data[box_size * count + 5] = 0;
                //top_data[box_size * count + 6] = i;
            }
	    count++;
    }
    assert(count == batch_num);
}


#ifdef DEBUG
int TestLayer::enqueue(int batchSize, const void *const* inputs, void **output, void *, cudaStream_t stream) {
    printf("\ntest layer input\n");
    float * tmp = (float*)malloc(sizeof(float)*count);
    CHECK(cudaMemcpy(tmp, (float*)inputs[0], sizeof(float) * count, cudaMemcpyDeviceToHost));
    for (int i = 0; i < 30; i++) {
        printf("%.10f ", tmp[i]);
    }
    printf("\n");
    free(tmp);
}


void TestLayer::configure(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) {
    count = inputs->d[0] * inputs->d[1]* inputs->d[2];
    printf("TestLayer Input CHW (%d %d %d)\n", inputs->d[0], inputs->d[1], inputs->d[2]);
    printf("TestLayer Output CHW (%d %d %d)\n", outputs->d[0], outputs->d[1], outputs->d[2]);
}

Dims TestLayer::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) {
    assert(nbInputDims == 1);
    return DimsCHW(32, 112, 208);
}

TestLayer::TestLayer(const nvinfer1::Weights* weights, int nbWeights) {
    printf("create testLayer\n");
    printf("nbWeights :%d count : %d\n", nbWeights, weights->count);
    printf("weight:\n");
    //float* tmp = (float*)malloc(sizeof(float)*weights->count);
    //cudaMemcpy(tmp, (float*)weights->values, sizeof(float)*weights->count, cudaMemcpyDeviceToHost);
    for(int i = 0; i < weights->count; i++ ) {
        printf("%.10f ", ((float*)(weights->values))[i]);
        if(((float*)(weights->values))[i] == (float)0)
            printf("zero");
    }
    printf("\n");
}
TestLayer::TestLayer(const void *data, size_t length){
    printf("create testLayer\n");
}
#endif
