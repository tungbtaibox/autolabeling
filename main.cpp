#include <iostream>
#include <QString>
#include <QDir>
#include "utils/autolabeling.h"

int main() {
    // ===== HARD CODE CÁC THAM SỐ Ở ĐÂY =====

    // Đường dẫn đến file model ZIP
    QString modelZipPath = "../../Model/Model.zip";  // Thay đổi đường dẫn này

    // Thư mục chứa ảnh cần gán nhãn
    QString imagesFolderPath = "../../images";  // Thay đổi đường dẫn này

    // Thư mục xuất kết quả (để trống để lưu cùng thư mục ảnh)
    QString outputDir = "../../assets";  // Thay đổi đường dẫn này

    // Các tham số ngưỡng
    float confThreshold = 0.6f;   // Ngưỡng confidence
    float iouThreshold = 0.45f;   // Ngưỡng IoU
    float maskThreshold = 0.5f;   // Ngưỡng mask



    try {
        // Tạo thư mục output nếu chưa có
        if (!outputDir.isEmpty()) {
            QDir().mkpath(outputDir);
        }

        // Thực hiện auto labeling
        int processedCount = AutoLabeling::processModelAndImages(
            modelZipPath,
            imagesFolderPath,
            outputDir,
            confThreshold,
            iouThreshold,
            maskThreshold
            );

        if (processedCount > 0) {
            std::cout << std::endl;
            std::cout << "Successfully processed " << processedCount << " images!" << std::endl;
            std::cout << "Check the output directory for results." << std::endl;
        } else {
            std::cerr << "No images were processed successfully." << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
