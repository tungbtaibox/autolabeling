#ifndef AUTOLABELING_H
#define AUTOLABELING_H

#include <string>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include "nn/autobackend.h" 
#include <unordered_map>
#include <QString>
#include <QDir>

struct ClassInfo {
    int ID;
    std::string ClassName;
    std::string Colour;
    int Items;
};

class AutoLabeling {
public:
    // Hàm chính để xử lý toàn bộ quy trình
    static int processModelAndImages(
        const QString& model_zip_path,
        const QString& images_folder_path,
        const QString& output_dir = "",
        float conf_threshold = 0.6f,
        float iou_threshold = 0.45f,
        float mask_threshold = 0.5f
    );

    // Hàm cũ giữ lại để tương thích
    static int autoLabelImages(
        const std::vector<std::string>& image_paths,
        const std::string& model_path,
        const std::string& output_dir = "",
        float conf_threshold = 0.6f,
        float iou_threshold = 0.45f,
        float mask_threshold = 0.5f
    );
    
    // Hàm overload mới để truyền class names từ bên ngoài
    static int autoLabelImages(
        const std::vector<std::string>& image_paths,
        const std::string& model_path,
        const std::unordered_map<int, std::string>& class_names,
        const std::string& output_dir = "",
        float conf_threshold = 0.6f,
        float iou_threshold = 0.45f,
        float mask_threshold = 0.5f
    );

private:
    static bool saveLabelMap(const std::string& output_path, 
                              const std::unordered_map<int, std::string>& names,
                              const std::map<int, int>& class_counts);
    static bool saveBBoxTxt(const std::string& txt_path, const std::vector<YoloResults>& results);
    static bool processSingleImage(
        const std::string& image_path,
        const std::string& model_path,
        const std::string& output_dir,
        float conf_threshold,
        float iou_threshold,
        float mask_threshold,
        std::map<int, int>& class_counts);
    
    // Các hàm helper mới
    static bool extractModelZip(const QString& zip_path, QString& extracted_path);
    static std::unordered_map<int, std::string> loadClassNamesFromExtractedModel(const QString& extracted_path);
    static std::string findOnnxModelInDirectory(const QString& directory_path);
    static std::vector<std::string> getImageFilesFromFolder(const QString& folder_path);
};

#endif // AUTOLABELING_H