#ifndef YOLOV8SEG_H
#define YOLOV8SEG_H

#include <QMainWindow>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/videoio.hpp>

#include <net.h>
#include <benchmark.h>
#include <cpu.h>

#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QMessageBox>
#include <QSize>
#include <QPixmap>
#include <QTime>
#include <QDateTime>
#include <QString>
#include <QMap>
#include <QVector>
#include <QTextStream>
#include <QList>
#include <QDebug>

// 中文乱码
#if _MSC_VER >= 1600
#pragma execution_character_set("utf-8")
#endif

// 定义实例分割数据结构
struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
    cv::Mat mask;
    std::vector<float> mask_feat;
};

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};


QT_BEGIN_NAMESPACE
namespace Ui { class yolov8Seg; }
QT_END_NAMESPACE

class yolov8Seg : public QMainWindow
{
    Q_OBJECT

public:
    yolov8Seg(QWidget *parent = nullptr);
    ~yolov8Seg();

    QImage cvMatToQImage(const cv::Mat& inMat);
    void DetectImage(int height, int width);

    int draw(cv::Mat& rgb, const std::vector<Object>& objects);

private slots:
    void on_btn_open_image_clicked();
    void on_btn_open_model_clicked();
    void on_btn_video_clicked();
    void on_btn_close_clicked();
    void on_btn_detect_clicked();


private:
    Ui::yolov8Seg *ui;

    QString modelPath;
    QList<QString> mClasses;
    QMap<int, QString> indexMapName;

    ncnn::Net yolo;
    cv::Mat Image;
    std::vector<Object> objects;

    const int target_size = 320;
    const float mean_vals[3] = { 103.53f, 116.28f, 123.675f };
    const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    float prob_threshold; // = 0.4f;
    float nms_threshold;  //  = 0.5f;
    const bool use_gpu = false;

    int image_w;
    int image_h;
    int in_w;
    int in_h;

    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;

    bool is_running;  // 用于跳出视频推理
};
#endif // YOLOV8SEG_H
