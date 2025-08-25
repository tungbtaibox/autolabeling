#include <iostream>
#include <QString>
#include <QDir>
#include "utils/autolabeling.h"

int main() {
    std::cout << "Testing ONNX Provider..." << std::endl;
    
    // Test với provider rỗng
    QString modelZipPath = "model.zip";
    QString imagesFolderPath = "./images";
    QString outputDir = "./output";
    
    try {
        // Tạo thư mục output nếu chưa có
        if (!outputDir.isEmpty()) {
            QDir().mkpath(outputDir);
        }
        
        std::cout << "Starting test with empty provider..." << std::endl;
        
        int processedCount = AutoLabeling::processModelAndImages(
            modelZipPath,
            imagesFolderPath,
            outputDir,
            0.6f,  // conf_threshold
            0.45f, // iou_threshold
            0.5f   // mask_threshold
        );
        
        if (processedCount > 0) {
            std::cout << "✓ SUCCESS: Processed " << processedCount << " images!" << std::endl;
        } else {
            std::cout << "✗ FAILED: No images were processed" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "✗ ERROR: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
