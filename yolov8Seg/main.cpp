#include "yolov8seg.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    yolov8Seg w;
    w.show();

    return a.exec();
}
