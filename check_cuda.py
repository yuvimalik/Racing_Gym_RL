import torch


def main():
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("Device count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}:", torch.cuda.get_device_name(i))
    else:
        print("No CUDA devices detected.")


if __name__ == "__main__":
    main()
