import torch


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Found GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("Torch works, but no GPU was found.")


if __name__ == '__main__':
    main()
