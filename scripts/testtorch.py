import torch
print(torch.__version__)
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
#Additional Info when using cuda
if device.type == 'cuda':
    print("Device name: ", torch.cuda.get_device_name(0))
    print("Device properties:", torch.cuda.get_device_properties(0))
    print("Utilization:", torch.cuda.utilization(0))
    print('Memory Usage:')
    print('Memory usage:', round(torch.cuda.memory_usage(0)/1024**3,1), 'GB')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB') #pip install pynvml
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))

print(torch.rand(2,3).cuda())

# import tensorrt
# print(tensorrt.__version__)
# assert tensorrt.Builder(tensorrt.Logger())