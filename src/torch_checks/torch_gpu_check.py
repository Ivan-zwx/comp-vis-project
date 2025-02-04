import torch
print(torch.version.cuda)  # Should output "12.6"
print(torch.backends.cudnn.version())  # Should return a valid cuDNN version (e.g., 8904)
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.device_count())  # Should be 1 (your RTX 4070)
print(torch.cuda.get_device_name(0))  # Should output "NVIDIA RTX 4070"

# import torch
# print(torch.version.cuda)  # Should return "12.6" if CUDA is enabled
# print(torch.cuda.is_available())  # Should return True
