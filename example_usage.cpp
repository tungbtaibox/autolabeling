#include <iostream>
#include <QString>
#include <QDir>
#include "utils/autolabeling.h"

int main() {
    // ===== VÍ DỤ 1: Xử lý ảnh thú cưng =====
    std::cout << "=== VÍ DỤ 1: Xử lý ảnh thú cưng ===" << std::endl;
    
    QString modelZipPath = "models/pet_detection.zip";
    QString imagesFolderPath = "./images/pets";
    QString outputDir = "./output/pets";
    
    float confThreshold = 0.7f;   // Ngưỡng cao hơn cho thú cưng
    float iouThreshold = 0.45f;
    float maskThreshold = 0.5f;
    
    int result1 = AutoLabeling::processModelAndImages(
        modelZipPath, imagesFolderPath, outputDir,
        confThreshold, iouThreshold, maskThreshold
    );
    
    std::cout << "Kết quả: " << result1 << " ảnh được xử lý" << std::endl << std::endl;
    
    // ===== VÍ DỤ 2: Xử lý ảnh xe cộ =====
    std::cout << "=== VÍ DỤ 2: Xử lý ảnh xe cộ ===" << std::endl;
    
    modelZipPath = "models/vehicle_detection.zip";
    imagesFolderPath = "./images/vehicles";
    outputDir = "./output/vehicles";
    
    confThreshold = 0.5f;   // Ngưỡng thấp hơn cho xe cộ
    iouThreshold = 0.4f;
    maskThreshold = 0.3f;
    
    int result2 = AutoLabeling::processModelAndImages(
        modelZipPath, imagesFolderPath, outputDir,
        confThreshold, iouThreshold, maskThreshold
    );
    
    std::cout << "Kết quả: " << result2 << " ảnh được xử lý" << std::endl << std::endl;
    
    // ===== VÍ DỤ 3: Xử lý ảnh người =====
    std::cout << "=== VÍ DỤ 3: Xử lý ảnh người ===" << std::endl;
    
    modelZipPath = "models/person_detection.zip";
    imagesFolderPath = "./images/people";
    outputDir = "./output/people";
    
    confThreshold = 0.6f;
    iouThreshold = 0.5f;
    maskThreshold = 0.4f;
    
    int result3 = AutoLabeling::processModelAndImages(
        modelZipPath, imagesFolderPath, outputDir,
        confThreshold, iouThreshold, maskThreshold
    );
    
    std::cout << "Kết quả: " << result3 << " ảnh được xử lý" << std::endl << std::endl;
    
    // ===== TỔNG KẾT =====
    std::cout << "=== TỔNG KẾT ===" << std::endl;
    std::cout << "Tổng số ảnh đã xử lý: " << (result1 + result2 + result3) << std::endl;
    
    return 0;
}
