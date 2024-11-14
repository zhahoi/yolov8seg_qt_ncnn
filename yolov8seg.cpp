#include "yolov8seg.h"
#include "ui_yolov8seg.h"

#define MAX_STRIDE 32

static void slice(const ncnn::Mat& in, ncnn::Mat& out, int start, int end, int axis)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Crop");

    // set param
    ncnn::ParamDict pd;

    ncnn::Mat axes = ncnn::Mat(1);
    axes.fill(axis);
    ncnn::Mat ends = ncnn::Mat(1);
    ends.fill(end);
    ncnn::Mat starts = ncnn::Mat(1);
    starts.fill(start);
    pd.set(9, starts);// start
    pd.set(10, ends);// end
    pd.set(11, axes);//axes

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}

static void interp(const ncnn::Mat& in, const float& scale, const int& out_w, const int& out_h, ncnn::Mat& out)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Interp");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 2);// resize_type
    pd.set(1, scale);// height_scale
    pd.set(2, scale);// width_scale
    pd.set(3, out_h);// height
    pd.set(4, out_w);// width

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}

static void reshape(const ncnn::Mat& in, ncnn::Mat& out, int c, int h, int w, int d)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Reshape");

    // set param
    ncnn::ParamDict pd;

    pd.set(0, w);// start
    pd.set(1, h);// end
    if (d > 0)
        pd.set(11, d);//axes
    pd.set(2, c);//axes
    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}

static void sigmoid(ncnn::Mat& bottom)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Sigmoid");

    op->create_pipeline(opt);

    // forward

    op->forward_inplace(bottom, opt);
    op->destroy_pipeline(opt);

    delete op;
}

static float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

static float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}

static void matmul(const std::vector<ncnn::Mat>& bottom_blobs, ncnn::Mat& top_blob)
{
    ncnn::Option opt;
    opt.num_threads = 2;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("MatMul");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 0);// axis

    op->load_param(pd);

    op->create_pipeline(opt);
    std::vector<ncnn::Mat> top_blobs(1);
    op->forward(bottom_blobs, top_blobs, opt);
    top_blob = top_blobs[0];

    op->destroy_pipeline(opt);

    delete op;
}

static float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    //     #pragma omp parallel sections
    {
        //         #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        //         #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.width * faceobjects[i].rect.height;
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static void generate_proposals(std::vector<GridAndStride> grid_strides, const ncnn::Mat& pred, float prob_threshold, std::vector<Object>& objects)
{
    const int num_points = grid_strides.size(); //网格点数量
    const int num_class = 80;
    const int reg_max_1 = 16; //计算回归系数DFL的长度

    for (int i = 0; i < num_points; i++)
    {
        const float* scores = pred.row(i) + 4 * reg_max_1; //scores指向分类索引

        // find label with max score
        int label = -1;
        float score = -FLT_MAX;
        for (int k = 0; k < num_class; k++)
        {
            float confidence = scores[k];
            if (confidence > score)
            {
                label = k;
                score = confidence;
            }
        }
        float box_prob = sigmoid(score);
        if (box_prob >= prob_threshold)
        {
            ncnn::Mat bbox_pred(reg_max_1, 4, (void*)pred.row(i));
            {
                ncnn::Layer* softmax = ncnn::create_layer("Softmax");

                ncnn::ParamDict pd;
                pd.set(0, 1); // axis
                pd.set(1, 1);
                softmax->load_param(pd);

                ncnn::Option opt;
                opt.num_threads = 1;
                opt.use_packing_layout = false;

                softmax->create_pipeline(opt);

                softmax->forward_inplace(bbox_pred, opt);

                softmax->destroy_pipeline(opt);

                delete softmax;
            }

            float pred_ltrb[4];
            for (int k = 0; k < 4; k++)
            {
                float dis = 0.f;
                const float* dis_after_sm = bbox_pred.row(k);
                for (int l = 0; l < reg_max_1; l++)//模型输出与其依次点乘得到预测框的回归系数
                {
                    dis += l * dis_after_sm[l];
                }

                pred_ltrb[k] = dis * grid_strides[i].stride;
            }

            float pb_cx = (grid_strides[i].grid0 + 0.5f) * grid_strides[i].stride;
            float pb_cy = (grid_strides[i].grid1 + 0.5f) * grid_strides[i].stride;

            float x0 = pb_cx - pred_ltrb[0];
            float y0 = pb_cy - pred_ltrb[1];
            float x1 = pb_cx + pred_ltrb[2];
            float y1 = pb_cy + pred_ltrb[3];

            Object obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = x1 - x0;
            obj.rect.height = y1 - y0;
            obj.label = label;
            obj.prob = box_prob;
            obj.mask_feat.resize(32); //
            std::copy(pred.row(i) + 64 + num_class, pred.row(i) + 64 + num_class + 32, obj.mask_feat.begin());
            objects.push_back(obj);
        }
    }
}

static void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
    for (int i = 0; i < (int)strides.size(); i++)
    {
        int stride = strides[i];
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++)
        {
            for (int g0 = 0; g0 < num_grid_w; g0++)
            {
                GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}

static void decode_mask(const ncnn::Mat& mask_feat, const int& img_w, const int& img_h,
    const ncnn::Mat& mask_proto, const ncnn::Mat& in_pad, const int& wpad, const int& hpad,
    ncnn::Mat& mask_pred_result)
{
    ncnn::Mat masks;
    matmul(std::vector<ncnn::Mat>{mask_feat, mask_proto}, masks); // masks(1, 5760)
    sigmoid(masks);
    // NCNN_LOGE("masks.w = %d", masks.w);
    // NCNN_LOGE("masks.h = %d", masks.h);
    // NCNN_LOGE("masks.c = %d", masks.c);
    reshape(masks, masks, masks.h, in_pad.h / 4, in_pad.w / 4, 0);
    slice(masks, mask_pred_result, (wpad / 2) / 4, (in_pad.w - wpad / 2) / 4, 2);
    slice(mask_pred_result, mask_pred_result, (hpad / 2) / 4, (in_pad.h - hpad / 2) / 4, 1);
    interp(mask_pred_result, 4.0, img_w, img_h, mask_pred_result);
}



yolov8Seg::yolov8Seg(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::yolov8Seg)
{
    ui->setupUi(this);

    qDebug() << "OpenCV Version: " << CV_VERSION;
    qDebug() << "OpenCV Major Version: " << CV_VERSION_MAJOR;
    qDebug() << "OpenCV Minor Version: " << CV_VERSION_MINOR;
    //根据不同模型输入分辨率大小不同，设置图像分辨率
    ui->comboBox->addItem(QString("%1 x %2").arg(320).arg(320));
    ui->comboBox->addItem(QString("%1 x %2").arg(640).arg(640));
    ui->comboBox->addItem(QString("%1 x %2").arg(800).arg(800));

    this->prob_threshold = ui->spinbox_conf_threshold->value();
    this->nms_threshold = ui->spinbox_iou_threshold->value();

    // 分割
    // this->readClassesFile(":/new/prefix1/resources/yolov5_classes.txt");
    this->blob_pool_allocator.set_size_compare_ratio(0.f);
    this->workspace_pool_allocator.set_size_compare_ratio(0.f);
    this->yolo.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    this->yolo.opt = ncnn::Option();

#if NCNN_VULKAN
    this->yolo.opt.use_vulkan_compute = use_gpu;
#endif

    this->yolo.opt.num_threads = ncnn::get_big_cpu_count();
    this->yolo.opt.blob_allocator = &blob_pool_allocator;
    this->yolo.opt.workspace_allocator = &workspace_pool_allocator;

    // 设置初始按钮状态
    ui->btn_open_image->setEnabled(true);
    ui->btn_open_model->setEnabled(true);
    ui->btn_detect->setEnabled(false);
    ui->btn_video->setEnabled(false);
    ui->btn_close->setEnabled(true);
}

yolov8Seg::~yolov8Seg()
{
    delete ui;
    this->yolo.clear();
}

QImage yolov8Seg::cvMatToQImage(const cv::Mat& inMat) {
    switch (inMat.type()) {
        // 8-bit, 4 channel
    case CV_8UC4: {
        QImage image(
            inMat.data,//图像数据
            inMat.cols,//图像宽度
            inMat.rows,//图像高度
            static_cast<int>(inMat.step),//图像矩阵元素类型
            QImage::Format_ARGB32//图像的像素格式
        );
        return image;
    }

                // 8-bit, 3 channel
    case CV_8UC3: {
        QImage image(inMat.data,
            inMat.cols,
            inMat.rows,
            static_cast<int>(inMat.step),
            QImage::Format_RGB888);
        return image.rgbSwapped(); //彩色图像为RGB三通道，交换R和B通道
    }

                // 8-bit, 1 channel
    case CV_8UC1: {
        static QVector<QRgb> sColorTable;
        // 我们只需要初始化一次颜色表
        if (sColorTable.isEmpty()) {
            for (int i = 0; i < 256; ++i) {
                sColorTable.push_back(qRgb(i, i, i));
            }
        }
        QImage image(inMat.data,
            inMat.cols,
            inMat.rows,
            static_cast<int>(inMat.step),
            QImage::Format_Indexed8);
        image.setColorTable(sColorTable);
        return image;
    }
    default:
        break;
    }
    return QImage();
}


void yolov8Seg::on_btn_open_image_clicked() {
    QString image_path = QFileDialog::getOpenFileName(this, QString("打开文件"), ".", tr("All Files(*.*);;MP3 Files(*.mp3);;MP4 Files(*.mp4)"));
    if (image_path.isEmpty()) {
        QMessageBox::information(this, "提示", "未选择文件");
        return;
    }
    QFileInfo fileinformation(image_path);

    this->Image = cv::imread(image_path.toStdString());
    if (this->Image.empty()) {
        QMessageBox::information(this, "提示", "打开图像失败");
        return;
    }

    qDebug() << "img height = " << this->Image.rows << " img width = " << this->Image.cols;

    //将当前读取的图像缩放到和显示图像的布局大小相同
    QSize labSize = ui->label_image->size();
    //将当前的图像缩放至和给定label_image布局一样大小，便于显示
    cv::resize(this->Image, this->Image, cv::Size(labSize.rwidth(), labSize.rheight()));

    qDebug() << "label_image height = " << this->Image.rows << "label_image width = " << Image.cols;

    QImage image = this->cvMatToQImage(this->Image);
    ui->label_image->setPixmap(QPixmap::fromImage(image));
    ui->label_image->repaint();  // 强制刷新

    // 封锁其他按钮
    ui->btn_open_image->setEnabled(true);
    ui->btn_open_model->setEnabled(true);
    ui->btn_detect->setEnabled(false);
    ui->btn_video->setEnabled(false);
    ui->btn_close->setEnabled(true);
}


void yolov8Seg::on_btn_open_model_clicked() {
    QString model_path = QFileDialog::getOpenFileName(this, QString("打开文件"), ".", tr("All Files(*.*);;MP3 Files(*.mp3);;MP4 Files(*.mp4)"));
    if (model_path.isEmpty()) {
        QMessageBox::information(this, "提示", "未选择文件");
        return;
    }
    this->modelPath = model_path;

    //加载模型文件
    if (this->yolo.load_param(this->modelPath.toStdString().c_str())) {
        QMessageBox::information(this, "提示", "加载param模型文件失败");
        return;
    }
    qDebug() << "modelPath: " << this->modelPath;

    QString fileName = this->modelPath.split(".")[0];
    qDebug() << "fileName: " << fileName;
    if (this->yolo.load_model(QString(fileName + ".bin").toStdString().c_str())) {
        QMessageBox::information(this, "提示", "加载bin权重文件失败");
        return;
    }

    qDebug() << "param path: " << this->modelPath;
    qDebug() << "bin path: " << fileName + ".bin";

    // 封锁其他按钮
    ui->btn_open_image->setEnabled(false);
    ui->btn_open_model->setEnabled(false);
    ui->btn_detect->setEnabled(true);
    ui->btn_video->setEnabled(true);
    ui->btn_close->setEnabled(true);
}


void yolov8Seg::DetectImage(int height, int width) {
    //对加载的图像进行缩放(height,width)
    int img_w = this->Image.cols;
    int img_h = this->Image.rows;

    qDebug() << "Input Image shape:" << img_w << ", " << img_h;

    // pad to multiple of 32
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)width / w;
        w = width;
        h = h * scale;
    }
    else
    {
        scale = (float)height / h;
        h = height;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(this->Image.data, ncnn::Mat::PIXEL_BGR2RGB,
        img_w, img_h, w, h);

    // pad to target_size rectangle
    int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolo.create_extractor();

    ex.input("images", in_pad);

    ncnn::Mat out;
    ex.extract("output0", out);

    ncnn::Mat mask_proto;
    ex.extract("output1", mask_proto);

    std::vector<int> strides = { 8, 16, 32 }; // might have stride=64
    std::vector<GridAndStride> grid_strides;
    generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);

    std::vector<Object> proposals;
    std::vector<Object> objects8;
    generate_proposals(grid_strides, out, prob_threshold, objects8);
    proposals.insert(proposals.end(), objects8.begin(), objects8.end());

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    ncnn::Mat mask_feat = ncnn::Mat(32, count, sizeof(float));
    for (int i = 0; i < count; i++) {
        float* mask_feat_ptr = mask_feat.row(i);
        std::memcpy(mask_feat_ptr, proposals[picked[i]].mask_feat.data(), sizeof(float) * proposals[picked[i]].mask_feat.size());
    }

    ncnn::Mat mask_pred_result;
    decode_mask(mask_feat, img_w, img_h, mask_proto, in_pad, wpad, hpad, mask_pred_result);
    //    NCNN_LOGE()

    this->objects.clear();
    objects.resize(count);

    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;

        objects[i].mask = cv::Mat::zeros(img_h, img_w, CV_32FC1);
        cv::Mat mask = cv::Mat(img_h, img_w, CV_32FC1, (float*)mask_pred_result.channel(i));
        mask(objects[i].rect).copyTo(objects[i].mask(objects[i].rect));
    }

    qDebug() << "objects size: " << objects.size();
}


int yolov8Seg::draw(cv::Mat& rgb, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
    };

    // static const char* class_names[] = {"blur", "phone", "reflectLight", "reflection"};
    static const unsigned char colors[81][3] = {
            {56,  0,   255},
            {226, 255, 0},
            {0,   94,  255},
            {0,   37,  255},
            {0,   255, 94},
            {255, 226, 0},
            {0,   18,  255},
            {255, 151, 0},
            {170, 0,   255},
            {0,   255, 56},
            {255, 0,   75},
            {0,   75,  255},
            {0,   255, 169},
            {255, 0,   207},
            {75,  255, 0},
            {207, 0,   255},
            {37,  0,   255},
            {0,   207, 255},
            {94,  0,   255},
            {0,   255, 113},
            {255, 18,  0},
            {255, 0,   56},
            {18,  0,   255},
            {0,   255, 226},
            {170, 255, 0},
            {255, 0,   245},
            {151, 255, 0},
            {132, 255, 0},
            {75,  0,   255},
            {151, 0,   255},
            {0,   151, 255},
            {132, 0,   255},
            {0,   255, 245},
            {255, 132, 0},
            {226, 0,   255},
            {255, 37,  0},
            {207, 255, 0},
            {0,   255, 207},
            {94,  255, 0},
            {0,   226, 255},
            {56,  255, 0},
            {255, 94,  0},
            {255, 113, 0},
            {0,   132, 255},
            {255, 0,   132},
            {255, 170, 0},
            {255, 0,   188},
            {113, 255, 0},
            {245, 0,   255},
            {113, 0,   255},
            {255, 188, 0},
            {0,   113, 255},
            {255, 0,   0},
            {0,   56,  255},
            {255, 0,   113},
            {0,   255, 188},
            {255, 0,   94},
            {255, 0,   18},
            {18,  255, 0},
            {0,   255, 132},
            {0,   188, 255},
            {0,   245, 255},
            {0,   169, 255},
            {37,  255, 0},
            {255, 0,   151},
            {188, 0,   255},
            {0,   255, 37},
            {0,   255, 0},
            {255, 0,   170},
            {255, 0,   37},
            {255, 75,  0},
            {0,   0,   255},
            {255, 207, 0},
            {255, 0,   226},
            {255, 245, 0},
            {188, 255, 0},
            {0,   255, 18},
            {0,   255, 75},
            {0,   255, 151},
            {255, 56,  0},
            {245, 255, 0}
    };

    int color_index = 0;
    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];
        const unsigned char* color = colors[color_index % 80];
        color_index++;

        cv::Scalar cc(color[0], color[1], color[2]);

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
            obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
        for (int y = 0; y < rgb.rows; y++) {
            uchar* image_ptr = rgb.ptr(y);
            const float* mask_ptr = obj.mask.ptr<float>(y);
            for (int x = 0; x < rgb.cols; x++) {
                if (mask_ptr[x] >= 0.5)
                {
                    image_ptr[0] = cv::saturate_cast<uchar>(image_ptr[0] * 0.5 + color[2] * 0.5);
                    image_ptr[1] = cv::saturate_cast<uchar>(image_ptr[1] * 0.5 + color[1] * 0.5);
                    image_ptr[2] = cv::saturate_cast<uchar>(image_ptr[2] * 0.5 + color[0] * 0.5);
                }
                image_ptr += 3;
            }
        }
        cv::rectangle(rgb, obj.rect, cc, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
            cv::Scalar(255, 255, 255), -1);

        cv::putText(rgb, text, cv::Point(x, y + label_size.height),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    return 0;
}



void yolov8Seg::on_btn_detect_clicked() {
    if (this->Image.empty() || this->modelPath.isEmpty()) {
        QMessageBox::information(this, "提示", "未加载图像或者模型");
        return;
    }
    //开始预测时间
    QDateTime curr = QDateTime::currentDateTime();
    qint64 startTime = curr.toMSecsSinceEpoch();

    //得到当前模型输入的图像分辨率大小
    QString resolution = ui->comboBox->currentText();
    int height = resolution.split("x")[0].trimmed().toInt();
    int width = resolution.split("x")[1].trimmed().toInt();

    //绘制坐标框信息到原图中
    this->DetectImage(height, width);

    // 可视化
    this->draw(this->Image, this->objects);

    QImage image = this->cvMatToQImage(this->Image);

    ui->label_image->setPixmap(QPixmap::fromImage(image));

    //推理结束时间
    qint64 endTime = curr.toMSecsSinceEpoch();
    ui->label_time->setText(QString::number(endTime - startTime) + "ms");

    // 开始检测封锁其他按钮
    ui->btn_open_image->setEnabled(false);
    ui->btn_open_model->setEnabled(false);
    ui->btn_detect->setEnabled(true);
    ui->btn_video->setEnabled(false);
    ui->btn_close->setEnabled(true);
}


//opencv打开摄像头进行检测
void yolov8Seg::on_btn_video_clicked() {
    cv::VideoCapture cap;
    if (!cap.open(0)) { // 检查摄像头是否成功打开
        QMessageBox::information(this, "提示", "无法打开摄像头");
        return;
    }

    is_running = true;  // 启动推理
    int count_fps = 0;
    auto fps_start = std::chrono::high_resolution_clock::now(); // 用于FPS计算的起始时间
    const int fps_update_interval_ms = 1000; // 每秒更新一次FPS

    //得到当前模型输入的图像分辨率大小
    QString resolution = ui->comboBox->currentText();
    QStringList dims = resolution.split("x");
    if (dims.size() != 2) {
        QMessageBox::information(this, "错误", "分辨率格式不正确");
        cap.release();
        return;
    }
    int height = dims[0].toInt();
    int width = dims[1].toInt();

    //打开摄像头，在is_running为true时继续推理
    while (cap.isOpened() && is_running) {
        cv::Mat frame;
        //读取视频帧并判断读取是否成功
        if (!cap.read(frame)) {
            QMessageBox::information(this, "提示", "打开摄像头失败");
            break;  // 跳出循环
        }

        this->Image = frame.clone();

        // 进行推理
        QSize labSize = ui->label_image->size();
        int scaled_width = static_cast<int>(this->Image.cols * static_cast<double>(labSize.width()) / this->Image.rows);
        int scaled_height = labSize.height();
        if (scaled_width > labSize.width()) {
            scaled_width = labSize.width();
            scaled_height = static_cast<int>(this->Image.rows * static_cast<double>(scaled_width) / this->Image.cols);
        }
        cv::resize(this->Image, this->Image, cv::Size(scaled_width, scaled_height));

        this->DetectImage(height, width);
        this->draw(this->Image, this->objects);

        QImage image = this->cvMatToQImage(this->Image);
        ui->label_image->setPixmap(QPixmap::fromImage(image.scaled(labSize, Qt::KeepAspectRatio, Qt::SmoothTransformation)));

        // 计算FPS
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - fps_start).count() >= fps_update_interval_ms) {
            float fps_value = count_fps * 1000.0f / fps_update_interval_ms;
            ui->label_fps->setText(QString::number(fps_value, 'f', 2));
            fps_start = now;
            count_fps = 0;
        }
        else {
            count_fps++;
        }

        // 处理Qt事件以响应UI操作
        QCoreApplication::processEvents();
    }

    // 释放资源
    cap.release();
}



void yolov8Seg::on_btn_close_clicked() {
    is_running = false; // 停止推理
    this->close();
}

