#ifndef AUTOLABELING_H
#define AUTOLABELING_H

#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include "nn/autobackend.h"
#include <unordered_map>
#include <QString>

struct ClassInfo {
    int ID;
    std::string ClassName;
    std::string Colour;  // Changed from int to std::string
    int Items;
};

class AutoLabeling {
public:
    // Main function for processing model and images
    static int processModelAndImages(
        const QString& model_zip_path,
        const QString& images_folder_path,
        const QString& output_dir,
        float conf_threshold = 0.6f,
        float iou_threshold = 0.45f,
        float mask_threshold = 0.5f
        );

    // Main auto labeling function
    static int autoLabelImages(
        const std::vector<std::string>& image_paths,
        const std::string& model_path,
        const std::string& output_dir = "",
        float conf_threshold = 0.6f,
        float iou_threshold = 0.45f,
        float mask_threshold = 0.5f
        );

    // Overloaded version with custom class names
    static int autoLabelImages(
        const std::vector<std::string>& image_paths,
        const std::string& model_path,
        const std::unordered_map<int, std::string>& class_names,
        const std::string& output_dir,
        float conf_threshold = 0.6f,
        float iou_threshold = 0.45f,
        float mask_threshold = 0.5f
        );

private:
    // Model extraction and setup functions
    static bool extractModelZip(const QString& zip_path, QString& extracted_path);
    static std::unordered_map<int, std::string> loadClassNamesFromExtractedModel(const QString& extracted_path);
    static std::string findOnnxModelInDirectory(const QString& directory_path);
    static std::vector<std::string> getImageFilesFromFolder(const QString& folder_path);

    // Processing functions
    static bool processSingleImage(
        const std::string& image_path,
        const std::string& model_path,
        const std::string& output_dir,
        float conf_threshold,
        float iou_threshold,
        float mask_threshold,
        std::map<int, int>& class_counts
        );

    // Output functions
    static bool saveLabelMap(const std::string& output_path,
                             const std::unordered_map<int, std::string>& names,
                             const std::map<int, int>& class_counts);
    static bool saveBBoxTxt(const std::string& txt_path, const std::vector<YoloResults>& results, const cv::Size& image_size);
};

#endif // AUTOLABELING_H
