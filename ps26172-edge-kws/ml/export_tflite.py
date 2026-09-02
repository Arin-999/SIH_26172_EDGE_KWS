import os

def export_tflite_to_c(tflite_path, header_path):
    with open(tflite_path, "rb") as f:
        tflite_model = f.read()
    
    os.makedirs(os.path.dirname(header_path), exist_ok=True)
    with open(header_path, "w") as f:
        f.write("#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n")
        f.write(f"const unsigned char g_model[] = {{\n")
        
        for i, byte in enumerate(tflite_model):
            f.write(f"0x{byte:02x}, ")
            if (i + 1) % 12 == 0:
                f.write("\n")
                
        f.write(f"\n}};\n")
        f.write(f"const unsigned int g_model_len = {len(tflite_model)};\n\n")
        f.write("#endif // MODEL_DATA_H\n")
    print(f"Exported {tflite_path} to {header_path}")

if __name__ == "__main__":
    export_tflite_to_c("ml/model_quantized.tflite", "firmware/ps26172_firmware/model_data.h")
