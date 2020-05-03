/*This is an application created to test the functionality of 
Deep Learning Image Super Resolution Model - Residual Dense Network in XILINX ZYNQ ULTRASCALE+ ZCU102 MPSoC
This code is considered opensource and accepts contributions and constructive criticism
Developed in Healthcare Technology Innovation Centre - Indian Institute of Technology, Madras

Further Upgradations
Unified Memory model for all functions
Portablilty to other applications eg :gstreamer
*/

#include <assert.h>
#include <dirent.h>
#include <dnndk/dnndk.h>
#include <dnndk/n2cube.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <cstring>


using namespace std;
using namespace std::chrono; //time

/* Specify Kernel name, input and output nodes as produced in compilation and 
load the kernel libraries to ZCU102 before running this application 
*/

#define DPU_KERNEL1 "RDN_44x44_C6D20G64G064x2"
#define DPU_KERNEL2 "RDN_84x84_C6D20G64G064x2"
#define DPU_KERNEL3 "RDN_124x124_C6D20G64G064x2"

#define INPUT_NODE "F_m1_convolution" //tf v1.12.0
#define OUTPUT_NODE "SR_convolution"  //tf v1.12.0

// Profiling time taken for a function 

#define SHOWTIME
#ifdef SHOWTIME
#define _T(func)                                                          \
  {                                                                       \
    auto _start = system_clock::now();                                    \
    func;                                                                 \
    auto _end = system_clock::now();                                      \
    auto duration = (duration_cast<microseconds>(_end - _start)).count(); \
    string tmp = #func;                                                   \
    tmp = tmp.substr(0, tmp.find('('));                                   \
    cout << "[TimeTest]" << left << setw(30) << tmp;                      \
    cout << left << setw(10) << duration << "us" << endl;                 \
  }
#else
#define _T(func) func;
#endif

void ListImages(std::string const &path, std::vector<std::string> &images) {
  images.clear();
  struct dirent *entry;

  /*Check if path is a valid directory path. */
  struct stat s;
  lstat(path.c_str(), &s);

  if (!S_ISDIR(s.st_mode)) {
    images.push_back(path);
    return;
  }

  DIR *dir = opendir(path.c_str());
  if (dir == nullptr) {
    fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
    exit(1);
  }

  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
      std::string name = entry->d_name;
      std::string ext = name.substr(name.find_last_of(".") + 1);
      if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") || (ext == "jpg") ||
          (ext == "PNG") || (ext == "png")) {
        images.push_back(path + "/" + name);
      }
    }
  }

  closedir(dir);
}

// setimage to DPU buffer

int dpuSetInputImage(DPUTask *task, const char* nodeName, const cv::Mat &image,int idx=0)
{
    int value;
    int8_t *inputAddr;
    unsigned char *resized_data;
    cv::Mat newImage;
    float scaleFix;
    int height, width, channel;

    height = dpuGetInputTensorHeight(task, nodeName, idx);
    width = dpuGetInputTensorWidth(task, nodeName, idx);
    channel = dpuGetInputTensorChannel(task, nodeName, idx);
    
    
    if (height == image.rows && width == image.cols) {
        newImage = image;
    }
    else{
        std::cout<<"Required image size "<<height<<"x"<<width<<"x"<<channel<<"\n";
        return -1;
    } 
    resized_data = newImage.data;

    inputAddr = dpuGetInputTensorAddress(task, nodeName, idx);
    scaleFix = dpuGetInputTensorScale(task, nodeName, idx);

    //possible issue
    for (int idx_h=0; idx_h<newImage.rows; idx_h++) {
            for (int idx_w=0; idx_w<newImage.cols; idx_w++) {
                for (int idx_c=0; idx_c<3; idx_c++) {
                  inputAddr[idx_h*newImage.cols*3+idx_w*3+idx_c] = newImage.at<cv::Vec3f>(idx_h, idx_w)[idx_c]* scaleFix; 
                }
            }
        }
    return scaleFix;
}

/*normalises input patch to 0-1,
  dpusetinputimage the data to buffer, 
  DPU processes the data and puts back to global buffer,
  getoutputtensor gets the data from global buffer to application memory
  rescale the pixel values to 0-255
*/
cv::Mat runPatch(cv::Mat patch,DPUTask *taskConv)
{
        int height=dpuGetOutputTensorHeight(taskConv,OUTPUT_NODE,0);
        int width=dpuGetOutputTensorWidth(taskConv,OUTPUT_NODE,0);
        int channel=dpuGetOutputTensorChannel(taskConv,OUTPUT_NODE,0);
        int total_size=height*width*channel;
        cv::Mat reimg= cv::Mat(height,width,CV_32FC3,cv::Scalar(0.0,0.0,0.0));

        int8_t outdata[total_size]={0};
        float scaleFix=0.0;

        cv::Mat image= patch;
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);//cvtcolor bgr-rgb
        cv::normalize(image,image,0,1,cv::NORM_MINMAX,CV_32F);///norm - must be present 32 -const
        scaleFix=dpuSetInputImage(taskConv,INPUT_NODE,image);//setimage--Flag issue 
        
        dpuRunTask(taskConv);
        
        dpuGetOutputTensorInHWCInt8(taskConv,OUTPUT_NODE,outdata,total_size);//get output
        
        //tensor to cv::mat
        
        for (int idx_h=0; idx_h<height; idx_h++) {
            for (int idx_w=0; idx_w<width; idx_w++) {
              for (int idx_c=0; idx_c<channel; idx_c++){
                  reimg.at<cv::Vec3f>(idx_h, idx_w)[2-idx_c]=outdata[idx_h*width*channel+idx_w*channel+idx_c]; 
                }
            }
        }
        cv::normalize(reimg,reimg,0,255,cv::NORM_MINMAX,-1);
        
        return reimg;

}


static int oheight;
static int owidth;

/*given an image matrix, this function splits the matrix into sub matrices of specified 2d size*/ 

std::vector<cv::Mat> patchify(cv::Mat img,int opatch_size,int padding_size=2)
{
 int patch_size=opatch_size;
 std::vector<cv::Mat> cpatches;
 cv::Mat patch;
 cv::Mat image= img;
 int width=image.cols;
 int height= image.rows;
 int w_rem= width % patch_size;
 int h_rem = height % patch_size;
 int w_extend = patch_size-w_rem;
 int h_extend = patch_size-h_rem;
 cv::Mat ext_image;
 cv::copyMakeBorder(image,ext_image,0,h_extend,0,w_extend,cv::BORDER_REPLICATE);
 cv::copyMakeBorder(ext_image,ext_image,padding_size,padding_size,padding_size,padding_size,cv::BORDER_REPLICATE);
 
 oheight=ext_image.rows;
 owidth=ext_image.cols;
 int w_left,w_width,h_top,h_height;
 for(int i=padding_size;i<ext_image.cols-padding_size;i+=patch_size){
     for(int j=padding_size;j<ext_image.rows-padding_size;j+=patch_size)
       {
         w_left= i-padding_size;
         h_top = j-padding_size;
         w_width = patch_size + 2*padding_size;
         h_height =  patch_size + 2*padding_size;
         cv::Rect crop(w_left,h_top,w_width,h_height);
         patch=ext_image(crop);
         cpatches.push_back(patch);
 }
}

return cpatches;

}
/*removes the padding added in patchify function*/
std::vector<cv::Mat> unpad(std::vector<cv::Mat> patches,int pad){
  cv::Mat patch;
  std::vector<cv::Mat> uppatches;
  for(auto it=patches.begin();it!=patches.end();++it)
  {
    patch=*it;
    uppatches.push_back(patch(cv::Rect(2*pad,2*pad,patch.cols-2*pad,patch.rows-2*pad))); 
  }
  return uppatches;
}

/*stiches the matrices back to specified size*/

cv::Mat depatchify(std::vector<cv::Mat> cpatches, int op_width, int op_height,int padding_size=4)
{
   cv::Mat image(2*oheight,2*owidth,CV_32FC3,cv::Scalar(0,0,0));
   std::vector<cv::Mat> patches=unpad(cpatches,padding_size);
   int patch_size=patches[0].cols;
   int patches_per_col = 2*oheight/patch_size;

   int col=-1,row=0;
   for(int i=0;i<patches.size();i++)
    {
       if (i % patches_per_col == 0)
        { ++col;
          row=0;
         }
       patches[i].copyTo(image(cv::Rect(col*patches[i].cols,row*patches[i].rows,patches[i].cols,patches[i].rows)));
    row++;}
   std::cout<<"rows:"<<image.rows<<"\n";
   std::cout<<"cols:"<<image.cols<<"\n";
   return image(cv::Rect(0,0,op_width,op_height));

}

/*main function*/

void runRDN_SR(std::string imgpath,DPUTask *taskConv,int patch_size){
    assert(taskConv);
    std::vector<cv::Mat> patches,patchesx2;
    cv::Mat patchx2;
  
    cv::Mat image= cv::imread(imgpath);
    int imwidth=image.cols;
    int imheight=image.rows;
    int errsize;
    std::cout<<"\n\nPROCESS STARTED\nProcessing "<<imgpath<<"\n";
    std::cout<<"width:"<<imwidth<<"\n";
    std::cout<<"height:"<<imheight<<"\n";
    
    int overlap=0;
    patches=patchify(image,patch_size,overlap);
  
    for(auto it=patches.begin();it!=patches.end();++it)
    {
        patchx2=runPatch(*it,taskConv);
        errsize=patchx2.cols;
        if(errsize!=2*patch_size)
          break;
        patchesx2.push_back(patchx2);    
    }

    if(patchesx2.size()==patches.size()){
        std::cout<<"Processed "<<patches.size()<<" patches\n";
        std::cout<<"conversion complete\n";
        std::cout<<"oheight:"<<oheight<<"\n";
        std::cout<<"owidth:"<<owidth<<"\n";
        cv::Mat HR=depatchify(patchesx2,2*imwidth,2*imheight,2*overlap);
        std::string name = imgpath.substr(imgpath.find_last_of("/") + 1);
        std::string save="HR"+std::to_string(patch_size)+"_"+std::to_string(overlap)+"_"+name;
        cv::imwrite(save,HR);
        std::cout<<"image written successfully as "<<save<<"\n";
    }
    else{
        std::cout<<"inconsistent patches!!\n output patch size "<<errsize<<"\n";
    } 
    patches.clear();
    patchesx2.clear();
}

/*boiler plate*/

void RDN_SR(const char* kernel,std::string ipath)
{
  DPUKernel *kernelConv;
  DPUTask *taskConv;
  
  int ipsize;

  if(kernel[4]=='4'){
    ipsize=44;
  }
  else if(kernel[4]=='8'){
    ipsize=84;
  }
  else if(kernel[4]=='1'){
    ipsize=124;
  }

  kernelConv = dpuLoadKernel(kernel);
  taskConv = dpuCreateTask(kernelConv, 0);
  vector<std::string> images;
  ListImages(ipath, images);
  for (std::string &filepath : images)
  {
    _T(runRDN_SR(filepath, taskConv,ipsize));
  }
  dpuDestroyTask(taskConv);
  dpuDestroyKernel(kernelConv);
  
}


int main(int argc, char** argv)
{
  if (argc != 2) {
    cout << "Usage of RDN: ./RDN_SR file_name" << endl;
    cout << "\tfile_name: path to your image file" << endl;
    return -1;
  }
  dpuOpen();
  RDN_SR(DPU_KERNEL1,argv[1]);
  RDN_SR(DPU_KERNEL2,argv[1]);
  RDN_SR(DPU_KERNEL3,argv[1]);
  dpuClose();
  
  return 0;
}

