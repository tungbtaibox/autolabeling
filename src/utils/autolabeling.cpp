#include "utils/autolabeling.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <utils/visualization.h> // Thêm include này để có hàm plot_results
#include <QProcess>
#include <QFileInfo>
#include <QDebug>
#include <QDir>
#include "constants.h"


using json = nlohmann::json;

// Hàm mới để giải nén model zip
bool AutoLabeling::extractModelZip(const QString& zip_path, QString& extracted_path) {
    QFileInfo fileInfo(zip_path);
    if (!fileInfo.exists()) {
        std::cerr << "Error: Zip file does not exist: " << zip_path.toStdString() << std::endl;
        return false;
    }

    // Tạo thư mục giải nén (sử dụng tên file zip không có extension)
    QString baseName = fileInfo.baseName();
    extracted_path = fileInfo.absolutePath() + "/" + baseName;

    QDir extractDir(extracted_path);
    if (extractDir.exists()) {
        // Xóa thư mục cũ nếu tồn tại
        extractDir.removeRecursively();
    }

    if (!extractDir.mkpath(".")) {
        std::cerr << "Error: Cannot create extraction directory: " << extracted_path.toStdString() << std::endl;
        return false;
    }

    // Giải nén file zip
#ifdef Q_OS_WINDOWS
    QString program = "powershell";
    QStringList arguments;
    arguments << "-Command" << QString("Expand-Archive -Path '%1' -DestinationPath '%2' -Force").arg(zip_path, extracted_path);
#else
    QString program = "unzip";
    QStringList arguments;
    arguments << "-o" << zip_path << "-d" << extracted_path;
#endif

    QProcess process;
    process.start(program, arguments);
    process.waitForFinished(-1);

    if (process.exitCode() == 0) {
        std::cout << "Successfully extracted model to: " << extracted_path.toStdString() << std::endl;
        return true;
    } else {
        std::cerr << "Error extracting zip file: " << process.readAllStandardError().toStdString() << std::endl;
        return false;
    }
}
// Phiên bản an toàn hơn với kiểm tra quyền truy cập
static bool deleteFolderSafe(const QString &folderPath) {
    QDir dir(folderPath);
    
    // Kiểm tra folder có tồn tại không
    if (!dir.exists()) {
        qDebug() << "Folder không tồn tại:" << folderPath;
        return false;
    }
    
    // Kiểm tra quyền ghi
    QFileInfo info(folderPath);
    if (!info.isWritable()) {
        qDebug() << "Không có quyền ghi folder:" << folderPath;
        return false;
    }
    
    // Xóa folder và toàn bộ nội dung
    bool success = dir.removeRecursively();
    
    if (success) {
        qDebug() << "Xóa folder thành công:" << folderPath;
    } else {
        qDebug() << "Xóa folder thất bại:" << folderPath;
    }
    
    return success;
}
// Hàm mới để load class names từ thư mục Categories
std::unordered_map<int, std::string> AutoLabeling::loadClassNamesFromExtractedModel(const QString& extracted_path) {
    std::unordered_map<int, std::string> classNames;

    // Tìm thư mục Categories trong thư mục giải nén
    QDir extractedDir(extracted_path);
    QStringList categoriesDirs = extractedDir.entryList(QStringList() << "*Categories*", QDir::Dirs);

    QString categoriesPath;
    if (!categoriesDirs.isEmpty()) {
        categoriesPath = extracted_path + "/" + categoriesDirs.first();
    } else {
        // Nếu không tìm thấy thư mục có tên Categories, tìm thư mục con đầu tiên
        QStringList subDirs = extractedDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
        if (!subDirs.isEmpty()) {
            // Kiểm tra xem thư mục con có chứa file với format tên.ID không
            QString firstSubDir = subDirs.first();
            QDir subDirPath(extracted_path + "/" + firstSubDir);
            QFileInfoList files = subDirPath.entryInfoList(QDir::Files);

            bool hasClassFiles = false;
            for (const QFileInfo& file : files) {
                QString fileName = file.fileName();
                int dotIndex = fileName.lastIndexOf(".");
                if (dotIndex != -1) {
                    QString idPart = fileName.mid(dotIndex + 1);
                    bool ok;
                    idPart.toInt(&ok);
                    if (ok) {
                        hasClassFiles = true;
                        break;
                    }
                }
            }

            if (hasClassFiles) {
                categoriesPath = extracted_path + "/" + firstSubDir;
            } else {
                std::cerr << "Error: No valid class files found in subdirectories" << std::endl;
                return classNames;
            }
        } else {
            std::cerr << "Error: No subdirectories found in extracted model" << std::endl;
            return classNames;
        }
    }

    QDir categoriesDir(categoriesPath);
    if (!categoriesDir.exists()) {
        std::cerr << "Error: Categories directory does not exist: " << categoriesPath.toStdString() << std::endl;
        return classNames;
    }

    // Đọc tất cả file trong thư mục Categories
    QFileInfoList fileList = categoriesDir.entryInfoList(QDir::Files | QDir::Readable, QDir::Name);

    for (const QFileInfo& fileInfo : fileList) {
        QString fileName = fileInfo.fileName();
        int dotIndex = fileName.lastIndexOf(".");
        if (dotIndex != -1) {
            QString namePart = fileName.left(dotIndex); // Phần trước dấu . (tên lớp)
            QString idPart = fileName.mid(dotIndex + 1); // Phần sau dấu . (ID)
            bool ok;
            int classId = idPart.toInt(&ok);
            if (ok) {
                classNames[classId] = namePart.toStdString();
                std::cout << "Loaded class: ID=" << classId << ", Name=" << namePart.toStdString() << std::endl;
            } else {
                std::cerr << "Warning: Invalid ID in filename: " << fileName.toStdString() << std::endl;
            }
        }
    }

    if (classNames.empty()) {
        std::cerr << "Error: No class names loaded from: " << categoriesPath.toStdString() << std::endl;
    } else {
        std::cout << "Successfully loaded " << classNames.size() << " class names" << std::endl;
    }

    return classNames;
}

// Hàm mới để tìm file ONNX model trong thư mục
std::string AutoLabeling::findOnnxModelInDirectory(const QString& directory_path) {
    QDir dir(directory_path);
    QStringList filters = {"*.onnx"};
    QStringList fileList = dir.entryList(filters, QDir::Files);

    if (fileList.isEmpty()) {
        // Tìm trong các thư mục con
        QStringList subDirs = dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
        for (const QString& subDir : subDirs) {
            QDir subDirPath(directory_path + "/" + subDir);
            QStringList subFileList = subDirPath.entryList(filters, QDir::Files);
            if (!subFileList.isEmpty()) {
                return (directory_path + "/" + subDir + "/" + subFileList.first()).toStdString();
            }
        }
        std::cerr << "Error: No ONNX model found in directory: " << directory_path.toStdString() << std::endl;
        return "";
    }

    return (directory_path + "/" + fileList.first()).toStdString();
}

// Hàm mới để lấy danh sách file ảnh từ thư mục
std::vector<std::string> AutoLabeling::getImageFilesFromFolder(const QString& folder_path) {
    std::vector<std::string> imageFiles;

    QDir dir(folder_path);
    if (!dir.exists()) {
        std::cerr << "Error: Images folder does not exist: " << folder_path.toStdString() << std::endl;
        return imageFiles;
    }

    QStringList filters = {"*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"};
    QStringList fileList = dir.entryList(filters, QDir::Files);

    if (fileList.isEmpty()) {
        std::cerr << "Error: No image files found in folder: " << folder_path.toStdString() << std::endl;
        return imageFiles;
    }

    for (const QString& fileName : fileList) {
        QString filePath = dir.filePath(fileName);
        imageFiles.push_back(filePath.toStdString());
    }

    std::cout << "Found " << imageFiles.size() << " image files" << std::endl;
    return imageFiles;
}

// Hàm chính mới để xử lý toàn bộ quy trình
int AutoLabeling::processModelAndImages(
    const QString& model_zip_path,
    const QString& images_folder_path,
    const QString& output_dir,
    float conf_threshold,
    float iou_threshold,
    float mask_threshold) {
    int autolabeling_result;
    try {
        std::cout << "Starting auto labeling process..." << std::endl;

        // Bước 1: Giải nén model zip
        QString extracted_path;
        if (!extractModelZip(model_zip_path, extracted_path)) {
            std::cerr << "Failed to extract model zip" << std::endl;
            return 0;
        }

        // Bước 2: Load class names từ thư mục Categories
        std::unordered_map<int, std::string> classNames = loadClassNamesFromExtractedModel(extracted_path);
        if (classNames.empty()) {
            std::cerr << "Failed to load class names" << std::endl;
            return 0;
        }

        // Bước 3: Tìm file ONNX model
        std::string model_path = findOnnxModelInDirectory(extracted_path);
        if (model_path.empty()) {
            std::cerr << "Failed to find ONNX model" << std::endl;
            return 0;
        }

        // Bước 4: Lấy danh sách file ảnh
        std::vector<std::string> imageFiles = getImageFilesFromFolder(images_folder_path);
        if (imageFiles.empty()) {
            std::cerr << "No image files found" << std::endl;
            return 0;
        }

        // Bước 5: Thực hiện auto labeling với class names từ Categories
        autolabeling_result = autoLabelImages(imageFiles, model_path, classNames, output_dir.toStdString(),
                               conf_threshold, iou_threshold, mask_threshold);
        if (!autolabeling_result) {
        //Bước 6: Xoa folder da giải nén
            if (!deleteFolderSafe(extracted_path)) {
                std::cerr << "Failed to delete extracted folder" << std::endl;
                return 0;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in processModelAndImages: " << e.what() << std::endl;
        return 0;
    }
    return autolabeling_result ;
}


bool AutoLabeling::saveLabelMap(const std::string& output_path,
                                const std::unordered_map<int, std::string>& names,
                                const std::map<int, int>& class_counts) {
    try {
        std::vector<ClassInfo> class_infos;

        for (const auto& [class_id, class_name] : names) {
            ClassInfo info;
            info.ID = class_id;
            info.ClassName = class_name;
            info.Colour = ""; // Empty string as requested

            // Get actual item count from class_counts, if not found then set to 0
            auto it = class_counts.find(class_id);
            info.Items = (it != class_counts.end()) ? it->second : 0;

            class_infos.push_back(info);
        }

        json j;
        for (const auto& info : class_infos) {
            json class_json;
            class_json["ID"] = info.ID;
            class_json["ClassName"] = info.ClassName;
            class_json["Colour"] = info.Colour;
            class_json["Items"] = info.Items;
            j.push_back(class_json);
        }

        std::ofstream file(output_path);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file for writing: " << output_path << std::endl;
            return false;
        }

        file << j.dump(4);
        file.close();

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving label map: " << e.what() << std::endl;
        return false;
    }
}

bool AutoLabeling::saveBBoxTxt(const std::string& txt_path, const std::vector<YoloResults>& results, const cv::Size& image_size) {
    try {
        std::ofstream file(txt_path);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file for writing: " << txt_path << std::endl;
            return false;
        }

        float img_width = static_cast<float>(image_size.width);
        float img_height = static_cast<float>(image_size.height);

        for (const auto& result : results) {
            // Tính toán tọa độ center và kích thước
            float x_center = result.bbox.x + result.bbox.width / 2.0f;
            float y_center = result.bbox.y + result.bbox.height / 2.0f;

            // Chuẩn hóa tọa độ về khoảng [0,1] theo quy định YOLO
            float normalized_x_center = x_center / img_width;
            float normalized_y_center = y_center / img_height;
            float normalized_width = result.bbox.width / img_width;
            float normalized_height = result.bbox.height / img_height;

            // Đảm bảo tọa độ nằm trong khoảng [0,1]
            normalized_x_center = std::max(0.0f, std::min(1.0f, normalized_x_center));
            normalized_y_center = std::max(0.0f, std::min(1.0f, normalized_y_center));
            normalized_width = std::max(0.0f, std::min(1.0f, normalized_width));
            normalized_height = std::max(0.0f, std::min(1.0f, normalized_height));

            // Lưu theo format YOLO: classId x_center y_center width height
            file << result.class_idx << " "
                 << std::fixed << std::setprecision(6)
                 << normalized_x_center << " " << normalized_y_center << " "
                 << normalized_width << " " << normalized_height;

            // Thêm keypoints nếu có (cũng chuẩn hóa)
            if (!result.keypoints.empty()) {
                for (size_t i = 0; i < result.keypoints.size(); i += 3) {
                    if (i + 1 < result.keypoints.size()) {
                        float kpt_x = result.keypoints[i] / img_width;
                        float kpt_y = result.keypoints[i + 1] / img_height;
                        float kpt_conf = (i + 2 < result.keypoints.size()) ? result.keypoints[i + 2] : 0.0f;

                        // Đảm bảo tọa độ keypoint nằm trong khoảng [0,1]
                        kpt_x = std::max(0.0f, std::min(1.0f, kpt_x));
                        kpt_y = std::max(0.0f, std::min(1.0f, kpt_y));

                        file << " " << std::fixed << std::setprecision(6)
                             << kpt_x << " " << kpt_y << " " << kpt_conf;
                    }
                }
            }

            file << "\n";
        }

        file.close();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving bbox txt: " << e.what() << std::endl;
        return false;
    }
}

bool AutoLabeling::processSingleImage(
    const std::string& image_path,
    const std::string& model_path,
    const std::string& output_dir,
    float conf_threshold,
    float iou_threshold,
    float mask_threshold,
    std::map<int, int>& class_counts) {

    try {
        if (!fs::exists(image_path)) {
            std::cerr << "Error: Image file does not exist: " << image_path << std::endl;
            return false;
        }

        if (!fs::exists(model_path)) {
            std::cerr << "Error: Model file does not exist: " << model_path << std::endl;
            return false;
        }

        fs::path output_path = output_dir.empty() ? fs::path(image_path).parent_path() : fs::path(output_dir);
        if (!output_dir.empty() && !fs::exists(output_path)) {
            fs::create_directories(output_path);
        }

        cv::Mat img = cv::imread(image_path, cv::IMREAD_UNCHANGED);
        if (img.empty()) {
            std::cerr << "Error: Unable to load image: " << image_path << std::endl;
            return false;
        }

        const std::string onnx_logid = "yolov8_autolabeling";
        const std::string onnx_provider = ""; // Để trống để sử dụng provider mặc định

        AutoBackendOnnx model(model_path.c_str(), onnx_logid.c_str(), onnx_provider.c_str());

        int conversion_code = cv::COLOR_BGR2RGB;
        std::vector<YoloResults> results = model.predict_once(
            img, conf_threshold, iou_threshold, mask_threshold, conversion_code
            );

        // Đếm số lượng objects cho mỗi class trong ảnh này
        for (const auto& result : results) {
            class_counts[result.class_idx]++;
        }

        fs::path image_stem = fs::path(image_path).stem();
        std::string output_stem = image_stem.string();

        std::vector<cv::Scalar> colors = generateRandomColors(model.getNc(), model.getCh());
        std::unordered_map<int, std::string> names = model.getNames(); // Giữ nguyên unordered_map

        cv::cvtColor(img, img, cv::COLOR_RGB2BGR);

        // Vẽ kết quả detection lên ảnh
        plot_results(img, results, colors, names, img.size());


        // Thêm thông tin từng class được detect
        int y_offset = 70;
        std::map<int, int> class_count;
        for (const auto& result : results) {
            class_count[result.class_idx]++;
        }

        for (const auto& [class_id, count] : class_count) {
            std::string class_name;
            auto it = names.find(class_id);
            if (it != names.end()) {
                class_name = it->second;
            } else {
                class_name = "Class_" + std::to_string(class_id);
            }

            std::string class_info = class_name + ": " + std::to_string(count);
            cv::putText(img, class_info, cv::Point(10, y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
            y_offset += 30;
        }

        std::string output_image_path = (output_path / (output_stem + "-labeled.jpg")).string();
        cv::imwrite(output_image_path, img);

        std::string output_txt_path = (output_path / (output_stem + ".txt")).string();
        if (!AutoLabeling::saveBBoxTxt(output_txt_path, results, img.size())) {
            std::cerr << "Error: Failed to save bbox txt file" << std::endl;
            return false;
        }

        std::cout << "Successfully processed: " << image_path << std::endl;
        std::cout << "Found " << results.size() << " objects" << std::endl;

        // In thông tin chi tiết về các objects được detect
        for (const auto& result : results) {
            std::string class_name;
            auto it = names.find(result.class_idx);
            if (it != names.end()) {
                class_name = it->second;
            } else {
                class_name = "Class_" + std::to_string(result.class_idx);
            }

            std::cout << "  - " << class_name << " (ID: " << result.class_idx
                      << ") at (" << result.bbox.x << "," << result.bbox.y
                      << ") size (" << result.bbox.width << "x" << result.bbox.height
                      << ") conf: " << std::fixed << std::setprecision(3) << result.conf << std::endl;
        }

        std::cout << "Output files:" << std::endl;
        std::cout << "  - Image: " << output_image_path << std::endl;
        std::cout << "  - BBox: " << output_txt_path << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error in processSingleImage: " << e.what() << std::endl;
        return false;
    }
}

int AutoLabeling::autoLabelImages(
    const std::vector<std::string>& image_paths,
    const std::string& model_path,
    const std::string& output_dir,
    float conf_threshold,
    float iou_threshold,
    float mask_threshold) {

    int success_count = 0;
    std::map<int, int> class_counts; // Map để đếm tổng số items của mỗi class

    // Khởi tạo model một lần để lấy danh sách class names
    const std::string onnx_logid = "yolov8_autolabeling";
    const std::string onnx_provider = ""; // Để trống để sử dụng provider mặc định
    AutoBackendOnnx model(model_path.c_str(), onnx_logid.c_str(), onnx_provider.c_str());
    std::unordered_map<int, std::string> names = model.getNames(); // Lấy class names từ model

    // Xử lý từng ảnh
    for (const auto& image_path : image_paths) {
        if (AutoLabeling::processSingleImage(image_path, model_path, output_dir,
                                             conf_threshold, iou_threshold, mask_threshold,
                                             class_counts)) {
            success_count++;
        }
    }

    // Sau khi xử lý tất cả ảnh, lưu file JSON với số lượng items thực tế
    fs::path output_path = output_dir.empty() ? fs::path(image_paths[0]).parent_path() : fs::path(output_dir);
    std::string output_json_path = (output_path / "ANS_Class.json").string();
    if (!AutoLabeling::saveLabelMap(output_json_path, names, class_counts)) {
        std::cerr << "Error: Failed to save label map JSON" << std::endl;
    } else {
        std::cout << "  - LabelMap: " << output_json_path << std::endl;
    }

    std::cout << "Processing completed. Successfully processed "
              << success_count << " out of " << image_paths.size()
              << " images." << std::endl;


    return success_count;
}

// Hàm overload mới để truyền class names từ bên ngoài
int AutoLabeling::autoLabelImages(
    const std::vector<std::string>& image_paths,
    const std::string& model_path,
    const std::unordered_map<int, std::string>& class_names,
    const std::string& output_dir,
    float conf_threshold,
    float iou_threshold,
    float mask_threshold) {

    int success_count = 0;
    std::map<int, int> class_counts; // Map để đếm tổng số items của mỗi class

    // Khởi tạo model một lần
    const std::string onnx_logid = "yolov8_autolabeling";
    const std::string onnx_provider = ""; // Để trống để sử dụng provider mặc định
    AutoBackendOnnx model(model_path.c_str(), onnx_logid.c_str(), onnx_provider.c_str());

    // Sử dụng class names được truyền vào thay vì từ model
    std::unordered_map<int, std::string> names = class_names;

    // Xử lý từng ảnh
    for (const auto& image_path : image_paths) {
        if (AutoLabeling::processSingleImage(image_path, model_path, output_dir,
                                             conf_threshold, iou_threshold, mask_threshold,
                                             class_counts)) {
            success_count++;
        }
    }

    // Sau khi xử lý tất cả ảnh, lưu file JSON với số lượng items thực tế
    fs::path output_path = output_dir.empty() ? fs::path(image_paths[0]).parent_path() : fs::path(output_dir);
    std::string output_json_path = (output_path / "ANS_Class.json").string();
    if (!AutoLabeling::saveLabelMap(output_json_path, names, class_counts)) {
        std::cerr << "Error: Failed to save label map JSON" << std::endl;
    } else {
        std::cout << "  - LabelMap: " << output_json_path << std::endl;
    }

    std::cout << "Processing completed. Successfully processed "
              << success_count << " out of " << image_paths.size()
              << " images." << std::endl;

    return success_count;
}


