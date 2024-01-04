import csv
import os
import platform
import psutil #pip install psutil #already installed
from datetime import datetime

import GPUtil #pip3 install gputil
from tabulate import tabulate #pip3 install tabulate
import torch

def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

def getLinuxReleaseinfo():
    RELEASE_DATA = {}
    with open("/etc/os-release") as f:
        reader = csv.reader(f, delimiter="=")
        for row in reader:
            if row:
                RELEASE_DATA[row[0]] = row[1]
    print(RELEASE_DATA)
    return RELEASE_DATA

def getlinuxCPUinfo():
    CPU_DATA = {}
    with open("/proc/cpuinfo") as f:
        reader = csv.reader(f, delimiter="=")
        for row in reader:
            if row:
                oneline=row[0].split(":")
                key=oneline[0].strip()
                if key in CPU_DATA.keys():
                    CPU_DATA[key].append(oneline[1])
                else:
                    CPU_DATA[key] =[oneline[1]]
    #print(CPU_DATA)
    totalcores=int(CPU_DATA['core id'][-1].strip())
    print("Total cores:",totalcores)
    totalthreads=len(CPU_DATA['core id']) #'processor'
    print("Total threads:",totalthreads)
    print("CPU Model:", CPU_DATA['model name'][0])
    return CPU_DATA, totalcores, totalthreads

def getSystemInfo():
    print("="*40, "System Information", "="*40)
    uname = platform.uname()
    print(f"System: {uname.system}")
    print(f"Node Name: {uname.node}")
    print(f"Release: {uname.release}")
    print(f"Version: {uname.version}")
    print(f"Machine: {uname.machine}")
    print(f"Processor: {uname.processor}")
    # Boot Time
    print("="*40, "Boot Time", "="*40)
    boot_time_timestamp = psutil.boot_time()
    bt = datetime.fromtimestamp(boot_time_timestamp)
    print(f"Boot Time: {bt.year}/{bt.month}/{bt.day} {bt.hour}:{bt.minute}:{bt.second}")

def getCPUinfo():
    # let's print CPU information
    print("="*40, "CPU Info", "="*40)
    # number of cores
    print("Physical cores:", psutil.cpu_count(logical=False))
    print("Total cores:", psutil.cpu_count(logical=True))
    # CPU frequencies
    cpufreq = psutil.cpu_freq()
    print(f"Max Frequency: {cpufreq.max:.2f}Mhz")
    print(f"Min Frequency: {cpufreq.min:.2f}Mhz")
    print(f"Current Frequency: {cpufreq.current:.2f}Mhz")
    # CPU usage
    print("CPU Usage Per Core:")
    for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
        print(f"Core {i}: {percentage}%")
    print(f"Total CPU Usage: {psutil.cpu_percent()}%")

def getMemoryInfo():
    # Memory Information
    print("="*40, "Memory Information", "="*40)
    # get the memory details
    svmem = psutil.virtual_memory()
    print(f"Total: {get_size(svmem.total)}")
    print(f"Available: {get_size(svmem.available)}")
    print(f"Used: {get_size(svmem.used)}")
    print(f"Percentage: {svmem.percent}%")
    print("="*20, "SWAP", "="*20)
    # get the swap memory details (if exists)
    swap = psutil.swap_memory()
    print(f"Total: {get_size(swap.total)}")
    print(f"Free: {get_size(swap.free)}")
    print(f"Used: {get_size(swap.used)}")
    print(f"Percentage: {swap.percent}%")

def getDiskInfo():
    # Disk Information
    print("="*40, "Disk Information", "="*40)
    print("Partitions and Usage:")
    # get all disk partitions
    partitions = psutil.disk_partitions()
    for partition in partitions:
        print(f"=== Device: {partition.device} ===")
        print(f"  Mountpoint: {partition.mountpoint}")
        print(f"  File system type: {partition.fstype}")
        try:
            partition_usage = psutil.disk_usage(partition.mountpoint)
        except PermissionError:
            # this can be catched due to the disk that
            # isn't ready
            continue
        print(f"  Total Size: {get_size(partition_usage.total)}")
        print(f"  Used: {get_size(partition_usage.used)}")
        print(f"  Free: {get_size(partition_usage.free)}")
        print(f"  Percentage: {partition_usage.percent}%")
    # get IO statistics since boot
    disk_io = psutil.disk_io_counters()
    print(f"Total read: {get_size(disk_io.read_bytes)}")
    print(f"Total write: {get_size(disk_io.write_bytes)}")

def getNetworkInfo():
    # Network information
    print("="*40, "Network Information", "="*40)
    # get all network interfaces (virtual and physical)
    if_addrs = psutil.net_if_addrs()
    for interface_name, interface_addresses in if_addrs.items():
        for address in interface_addresses:
            print(f"=== Interface: {interface_name} ===")
            if str(address.family) == 'AddressFamily.AF_INET':
                print(f"  IP Address: {address.address}")
                print(f"  Netmask: {address.netmask}")
                print(f"  Broadcast IP: {address.broadcast}")
            elif str(address.family) == 'AddressFamily.AF_PACKET':
                print(f"  MAC Address: {address.address}")
                print(f"  Netmask: {address.netmask}")
                print(f"  Broadcast MAC: {address.broadcast}")
    # get IO statistics since boot
    net_io = psutil.net_io_counters()
    print(f"Total Bytes Sent: {get_size(net_io.bytes_sent)}")
    print(f"Total Bytes Received: {get_size(net_io.bytes_recv)}")

def getGPUInfo():
    print("="*40, "GPU Details", "="*40)
    if torch.cuda.is_available():
        devicecount=torch.cuda.device_count()
        for i in range(devicecount):
            print("GPU device:", i)
            print("Device name: ", torch.cuda.get_device_name(i))
            print("Device properties:", torch.cuda.get_device_properties(i))
            print('Memory usage:', round(torch.cuda.memory_usage(i)/1024**3,1), 'GB')

        #gpulist=GPUtil.getAvailable()
        gpus = GPUtil.getGPUs()
        list_gpus = []
        for gpu in gpus:
            # get the GPU id
            gpu_id = gpu.id
            # name of GPU
            gpu_name = gpu.name
            # get % percentage of GPU usage of that GPU
            gpu_load = f"{gpu.load*100}%"
            # get free memory in MB format
            gpu_free_memory = f"{gpu.memoryFree}MB"
            # get used memory
            gpu_used_memory = f"{gpu.memoryUsed}MB"
            # get total memory
            gpu_total_memory = f"{gpu.memoryTotal}MB"
            # get GPU temperature in Celsius
            gpu_temperature = f"{gpu.temperature} °C"
            gpu_uuid = gpu.uuid
            list_gpus.append((
                gpu_id, gpu_name, gpu_load, gpu_free_memory, gpu_used_memory,
                gpu_total_memory, gpu_temperature, gpu_uuid
            ))

        print(tabulate(list_gpus, headers=("id", "name", "load", "free memory", "used memory", "total memory",
                                        "temperature", "uuid")))

def getDeviceType():
    is_windows = any(platform.win32_ver())
    platformname = platform.system() #Returns the system/OS name, e.g. 'Linux', 'Windows' or 'Java'.
    hostname=platform.node() #'coe-hpc1.sjsuad.sjsu.edu'
    is_hpc= platformname=="Linux" and hostname.startswith("coe-hpc")
    is_hpc = False
    is_hpc1gpu = False
    is_hpc2gpu = False
    if platformname=="Linux":
        RELEASE_DATA = getLinuxReleaseinfo()
        CPU_DATA, totalcores, totalthreads = getlinuxCPUinfo()
        if hostname.startswith("coe-hpc"):
            is_hpc = True
            print("HPC head node")
        elif RELEASE_DATA['NAME'].startswith("CentOS"):
            if hostname.startswith("g") or hostname.startswith("condo"):
                print("HPC1 GPU node")
                is_hpc = True
                is_hpc1gpu = True
            elif hostname.startswith("cs"):
                print("HPC2 GPU node")
                is_hpc = True
                is_hpc2gpu = True
    elif is_windows:
        print("Windows machine")
    return is_windows, is_hpc, is_hpc1gpu, is_hpc2gpu

if __name__ == "__main__":
    getSystemInfo()
    getCPUinfo()
    getMemoryInfo()
    getDiskInfo()
    getNetworkInfo()
    getGPUInfo()
    getDeviceType()
