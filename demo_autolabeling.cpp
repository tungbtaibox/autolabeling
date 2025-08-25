#include <iostream>
#include <QString>
#include <QDir>
#include <QFileInfo>
#include "utils/autolabeling.h"

void testExtractModelZip() {
    std::cout << "=== Testing Model ZIP Extraction ===" << std::endl;
    
    QString zipPath = "test_model.zip";
    QString extractedPath;
    
    if (AutoLabeling::extractModelZip(zipPath, extractedPath)) {
        std::cout << "✓ Successfully extracted to: " << extractedPath.toStdString() << std::endl;
    } else {
        std::cout << "✗ Failed to extract model ZIP" << std::endl;
    }
    std::cout << std::endl;
}

void testLoadClassNames() {
    std::cout << "=== Testing Class Names Loading ===" << std::endl;
    
    QString extractedPath = "test_model";
    auto classNames = AutoLabeling::loadClassNamesFromExtractedModel(extractedPath);
    
    if (!classNames.empty()) {
        std::cout << "✓ Loaded " << classNames.size() << " class names:" << std::endl;
        for (const auto& [id, name] : classNames) {
            std::cout << "  - ID: " << id << ", Name: " << name << std::endl;
        }
    } else {
        std::cout << "✗ Failed to load class names" << std::endl;
    }
    std::cout << std::endl;
}

void testFindOnnxModel() {
    std::cout << "=== Testing ONNX Model Finding ===" << std::endl;
    
    QString directoryPath = "test_model";
    std::string modelPath = AutoLabeling::findOnnxModelInDirectory(directoryPath);
    
    if (!modelPath.empty()) {
        std::cout << "✓ Found ONNX model: " << modelPath << std::endl;
    } else {
        std::cout << "✗ No ONNX model found" << std::endl;
    }
    std::cout << std::endl;
}

void testGetImageFiles() {
    std::cout << "=== Testing Image Files Loading ===" << std::endl;
    
    QString folderPath = "./images";
    auto imageFiles = AutoLabeling::getImageFilesFromFolder(folderPath);
    
    if (!imageFiles.empty()) {
        std::cout << "✓ Found " << imageFiles.size() << " image files:" << std::endl;
        for (const auto& file : imageFiles) {
            std::cout << "  - " << file << std::endl;
        }
    } else {
        std::cout << "✗ No image files found" << std::endl;
    }
    std::cout << std::endl;
}

void testFullProcess() {
    std::cout << "=== Testing Full Auto Labeling Process ===" << std::endl;
    
    QString modelZipPath = "test_model.zip";
    QString imagesFolderPath = "./images";
    QString outputDir = "./output";
    
    // Tạo thư mục output nếu chưa có
    QDir().mkpath(outputDir);
    
    int processedCount = AutoLabeling::processModelAndImages(
        modelZipPath,
        imagesFolderPath,
        outputDir,
        0.6f,  // conf_threshold
        0.45f, // iou_threshold
        0.5f   // mask_threshold
    );
    
    if (processedCount > 0) {
        std::cout << "✓ Successfully processed " << processedCount << " images!" << std::endl;
        std::cout << "Check output directory: " << outputDir.toStdString() << std::endl;
    } else {
        std::cout << "✗ No images were processed successfully" << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "Auto Labeling Tool - Demo" << std::endl;
    std::cout << "=========================" << std::endl;
    std::cout << std::endl;
    
    // Test từng chức năng riêng biệt
    testExtractModelZip();
    testLoadClassNames();
    testFindOnnxModel();
    testGetImageFiles();
    
    // Test toàn bộ quy trình
    testFullProcess();
    
    std::cout << "Demo completed!" << std::endl;
    return 0;
}
