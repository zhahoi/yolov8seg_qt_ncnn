#-------------------------------------------------
#
# Project created by QtCreator 2024-11-13T18:23:09
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

# OPENMP
QMAKE_CXXFLAGS += -fopenmp
QMAKE_LFLAGS += -fopenmp

TARGET = yolov8Seg
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


SOURCES += \
        main.cpp \
        yolov8seg.cpp

HEADERS += \
        yolov8seg.h

FORMS += \
        yolov8seg.ui

# opencv
INCLUDEPATH += /usr/local/include \
                /usr/local/include/opencv \
                /usr/local/include/opencv2 \

LIBS += /usr/local/lib/libopencv_*

# ncnn
INCLUDEPATH += /home/hit/Softwares/ncnn \
                /home/hit/Softwares/ncnn/build/install/include/ncnn

LIBS += -L/home/hit/Softwares/ncnn/build/install/lib -lncnn

# RESOURCES += \
#     res.qrc



