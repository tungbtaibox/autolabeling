# Auto Labeling Tool

Công cụ tự động gán nhãn ảnh sử dụng YOLO model từ file ZIP.

## Tính năng

1. **Giải nén model ZIP**: Tự động giải nén file model ZIP và tìm file ONNX model
2. **Load class names**: Đọc tên các class từ thư mục Categories trong model
3. **Auto labeling**: Tự động gán nhãn cho tất cả ảnh trong thư mục
4. **Xuất kết quả**: Tạo file TXT cho mỗi ảnh và file JSON tổng hợp

## Cấu trúc Model ZIP

Model ZIP cần có cấu trúc như sau:
```
model.zip
├── Categories/
│   ├── pet.1
│   ├── dog.2
│   ├── cat.3
│   └── ...
├── model.onnx
└── ...
```

- Thư mục `Categories/` chứa các file với format: `tên_class.ID`
- File `model.onnx` là model YOLO để inference

## Cách sử dụng

### Build project
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### Chạy auto labeling
```bash
# Chạy trực tiếp
./autolabeling_tool.exe
```

### Cấu hình tham số
Chỉnh sửa các tham số trong file `main_autolabeling.cpp`:

```cpp
// Đường dẫn đến file model ZIP
QString modelZipPath = "model.zip";  // Thay đổi đường dẫn này

// Thư mục chứa ảnh cần gán nhãn
QString imagesFolderPath = "./images";  // Thay đổi đường dẫn này

// Thư mục xuất kết quả
QString outputDir = "./output";  // Thay đổi đường dẫn này

// Các tham số ngưỡng
float confThreshold = 0.6f;   // Ngưỡng confidence
float iouThreshold = 0.45f;   // Ngưỡng IoU
float maskThreshold = 0.5f;   // Ngưỡng mask
```

## Định dạng file kết quả

### File TXT (cho mỗi ảnh)
Format: `classId-1 x_center y_center width height`
```
0 0.5 0.3 0.2 0.4
1 0.7 0.6 0.3 0.5
```

### File ANS_Class.json
```json
[
    {
        "ID": 1,
        "ClassName": "pet",
        "Colour": "",
        "Items": 4
    },
    {
        "ID": 2,
        "ClassName": "dog",
        "Colour": "",
        "Items": 2
    }
]
```

## Quy trình xử lý

1. **Giải nén model ZIP** → Tạo thư mục tạm
2. **Load class names** → Đọc từ thư mục Categories
3. **Tìm ONNX model** → Tìm file .onnx trong thư mục giải nén
4. **Load ảnh** → Quét tất cả file ảnh (.jpg, .png, .jpeg, .bmp, .tiff, .tif)
5. **Inference** → Chạy model YOLO trên từng ảnh
6. **Lưu kết quả** → Tạo file TXT và JSON
7. **Dọn dẹp** → Xóa thư mục tạm

## Yêu cầu hệ thống

- Windows 10/11
- Visual Studio 2019/2022
- Qt6
- OpenCV 4.x
- ONNX Runtime

## Lưu ý

- File ảnh hỗ trợ: .jpg, .jpeg, .png, .bmp, .tiff, .tif
- Model phải là YOLO format ONNX
- Class ID bắt đầu từ 1 (trong file TXT sẽ lưu classId-1)
- Kết quả được lưu theo format YOLO standard
