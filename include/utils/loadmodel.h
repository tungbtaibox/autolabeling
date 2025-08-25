#ifndef LOADMODEL
#define LOADMODEL


#include <nn/onnx_model_base.h>
#include <nn/autobackend.h>
#include <vector>
#include <QProcess>
#include <QFileInfo>
#include <QDebug>

#include <QDir>
#include <QString>
namespace fs = std::filesystem;

cv::Scalar generateRandomColor(int numChannels);
std::vector<cv::Scalar> generateRandomColors(int class_names_num, int numChannels);
void plot_results(cv::Mat img, std::vector<YoloResults>& results,
                  std::vector<cv::Scalar> color, std::unordered_map<int, std::string>& names,
                  const cv::Size& shape
                  );
bool extractAndDeleteZip(const QString& zipFilePath, const QString& extractDir);
std::vector<cv::Scalar> generateRandomColors(int class_names_num, int numChannels);
std::map<int, std::string> loadClassNamesFromCategories(const QString& categoryDir);
bool load_model(std::map<int, std::string>* classNamesMap,const QString& zipFileDir,const QString &folderImagesPath);
#endif
