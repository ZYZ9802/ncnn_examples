#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "platform.h"
#include "net.h"
using namespace std;
using namespace cv;

static int detect_squeezenet(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
    ncnn::Net squeezenet;

#if NCNN_VULKAN
    squeezenet.opt.use_vulkan_compute = true;
#endif // NCNN_VULKAN

    squeezenet.load_param("squeezenet_v1.1.param");
    squeezenet.load_model("squeezenet_v1.1.bin");

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 227, 227);

    const float mean_vals[3] = {104.f, 117.f, 123.f};
    in.substract_mean_normalize(mean_vals, 0);

    ncnn::Extractor ex = squeezenet.create_extractor();

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);

    cls_scores.resize(out.w);
    for (int j=0; j<out.w; j++)
    {
        cls_scores[j] = out[j];
    }

    return 0;
}
//添加load labels函数，读取synset_words.txt
static int load_labels(string path, std::vector<string>& labels)
{
    FILE* fp = fopen(path.c_str(), "r");
    
    while (!feof(fp))
    {
        char str[1024];
        fgets(str, 1024, fp);  
        string str_s(str);
        
        if (str_s.length() > 0)
        {
            for (int i = 0; i < str_s.length(); i++)
            {
                if (str_s[i] == ' ')
                {
                    string strr = str_s.substr(i, str_s.length() - i - 1);
                    labels.push_back(strr);
                    i = str_s.length();
                }
            }
        }
    }
    return 0;
}

static int print_topk(const cv::Mat& bgr,const std::vector<float>& cls_scores, int topk)
{
    //load labels 
    cv::Mat image = bgr.clone();
    vector<string> labels;
    vector<int> index_result;
    load_labels("synset_words.txt", labels);
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector< std::pair<float, int> > vec;
    vec.resize(size);
    for (int i=0; i<size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater< std::pair<float, int> >());

    // print topk and score
    for (int i=0; i<topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        //fprintf(stderr, "%d = %f\n", index, score);
	index_result.push_back(index);
	fprintf(stderr, "%d = %f (%s)\n", index, score, labels[index].c_str());
    }
   //在图片上标注类别信息
    for (int i = 0;i<index_result.size()-2;i++)
    {
       cv::putText(image, labels[index_result[i]], Point(50, 50 + 30 * i), CV_FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255,255), 2, 8);
    }

    imshow("image", image);
    imwrite("squezenet_new.jpg", image);
    waitKey(0);
    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }
    std::vector<float> cls_scores;
    detect_squeezenet(m, cls_scores);

    print_topk(m,cls_scores, 3);
     

    return 0;
}
