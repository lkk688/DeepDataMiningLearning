import argparse
import json
import itertools as it
import operator
import os
import shlex
import shutil
import subprocess
import sys


__version__ = "1.0.0"


NVIDIA_SMI_GET_GPUS = "nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.total,memory.used,memory.free,driver_version,name,gpu_serial,display_active,display_mode,temperature.gpu --format=csv,noheader,nounits"
r"""
The command is used to query specific GPU metrics and display them in a CSV (comma-separated values) format. It retrieves information related to GPU utilization, memory, driver version, and other relevant details.
**Parameters**:
   - **--query-gpu**: Specifies the GPU attributes to query. The comma-separated list includes:
     - **index**: The index of the GPU.
     - **uuid**: Universally unique identifier for the GPU.
     - **utilization.gpu**: Percent of time during which one or more kernels were executing on the GPU.
     - **memory.total**: Total installed GPU memory.
     - **memory.used**: Total memory allocated by active contexts.
     - **memory.free**: Total free memory.
     - **driver_version**: The version of the installed NVIDIA display driver.
     - **name**: The official product name of the GPU.
     - **gpu_serial**: Serial number of the GPU.
     - **display_active**: Whether the GPU is actively driving a display (1 for active, 0 for inactive).
     - **display_mode**: Current display mode (e.g., "Enabled", "Disabled").
     - **temperature.gpu**: Core GPU temperature in degrees Celsius.
   - **--format=csv,noheader,nounits**:
     - Specifies the output format as CSV.
     - **noheader**: Excludes the header row from the output.
     - **nounits**: Removes units (e.g., "MB", "%") from numeric values.

"""

NVIDIA_SMI_GET_PROCS = "nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,gpu_name,used_memory --format=csv,noheader,nounits"
r"""
monitoring GPU usage, identifying processes utilizing the GPU, and understanding memory consumption by different applications.
**Options**:
    - `--query-compute-apps=pid,process_name,gpu_uuid,gpu_name,used_memory`
        - This option specifies the query to retrieve specific information from the GPU.
        - The requested fields are:
            - `pid`: Process ID of the running application using the GPU.
            - `process_name`: Name of the process/application.
            - `gpu_uuid`: Unique identifier for the GPU.
            - `gpu_name`: Name of the GPU model.
            - `used_memory`: Amount of GPU memory (in megabytes) currently in use by the application.
    - `--format=csv,noheader,nounits`
        - This option specifies the output format for the query results.
        - `csv`: The output will be in CSV (Comma-Separated Values) format.
        - `noheader`: No header row will be included in the output.
        - `nounits`: The memory usage will be reported without units (e.g., just the numeric value).

**Example Output**:
    ```
    1234,my_app.exe,GPU-12345678-1234-5678-9012-345678901234,Tesla V100,2048
    5678,another_app.exe,GPU-98765432-9876-5432-2109-876543210987,Tesla P100,1024
    ```
    - In this example, two processes are running:
        - Process ID `1234` (named `my_app.exe`) is using a Tesla V100 GPU with 2048 MB of memory.
        - Process ID `5678` (named `another_app.exe`) is using a Tesla P100 GPU with 1024 MB of memory.

Refer to the [nvidia-smi documentation](https://developer.nvidia.com/nvidia-system-management-interface).

"""

class GPU(object):
    def __init__(
        self,
        id,
        uuid,
        gpu_util,
        mem_total,
        mem_used,
        mem_free,
        driver,
        gpu_name,
        serial,
        display_mode,
        display_active,
        temperature,
    ):
        self.id = id
        self.uuid = uuid
        self.gpu_util = gpu_util
        self.mem_util = float(mem_used) / float(mem_total) * 100
        self.mem_total = mem_total
        self.mem_used = mem_used
        self.mem_free = mem_free
        self.driver = driver
        self.name = gpu_name
        self.serial = serial
        self.display_mode = display_mode
        self.display_active = display_active
        self.temperature = temperature

    def __repr__(self):
        msg = "id: {id} | UUID: {uuid} | gpu_util: {gpu_util:5.1f}% | mem_util: {mem_util:5.1f}% | mem_free: {mem_free:7.1f}MB |  mem_total: {mem_total:7.1f}MB"
        msg = msg.format(**self.__dict__)
        return msg

    def to_json(self):
        return json.dumps(self.__dict__)


class GPUProcess(object):
    def __init__(self, pid, process_name, gpu_id, gpu_uuid, gpu_name, used_memory):
        self.pid = pid
        self.process_name = process_name
        self.gpu_id = gpu_id
        self.gpu_uuid = gpu_uuid
        self.gpu_name = gpu_name
        self.used_memory = used_memory

    def __repr__(self):
        msg = "pid: {pid} | gpu_id: {gpu_id} | gpu_uuid: {gpu_uuid} | gpu_name: {gpu_name} | used_memory: {used_memory:7.1f}MB"
        msg = msg.format(**self.__dict__)
        return msg

    def to_json(self):
        return json.dumps(self.__dict__)


def to_float_or_inf(value):
    try:
        number = float(value)
    except ValueError:
        number = float("nan")
    return number


def _get_gpu(line):
    values = line.split(", ")
    id = values[0]
    uuid = values[1]
    gpu_util = to_float_or_inf(values[2])
    mem_total = to_float_or_inf(values[3])
    mem_used = to_float_or_inf(values[4])
    mem_free = to_float_or_inf(values[5])
    driver = values[6]
    gpu_name = values[7]
    serial = values[8]
    display_active = values[9]
    display_mode = values[10]
    temp_gpu = to_float_or_inf(values[11])
    gpu = GPU(
        id,
        uuid,
        gpu_util,
        mem_total,
        mem_used,
        mem_free,
        driver,
        gpu_name,
        serial,
        display_mode,
        display_active,
        temp_gpu,
    )
    return gpu


def get_gpus():
    output = subprocess.check_output(shlex.split(NVIDIA_SMI_GET_GPUS))
    #check_output(): A function within the subprocess module that runs a command and captures its output.
    #shlex.split(NVIDIA_SMI_GET_GPUS): Splits the command string into a list of arguments, ensuring proper handling of spaces and special characters.
    #The output of the command (including details like GPU index, UUID, utilization, memory usage, etc.) is captured and stored in the output variable.

    lines = output.decode("utf-8").split(os.linesep)
    #decodes the output (which is a byte string) from UTF-8 encoding to a regular Unicode string.
    #It then splits the resulting string into a list of lines using the os.linesep separator (which represents the newline character, \n).
    #The variable lines now contains a list of individual lines from the decoded output.

    gpus = (_get_gpu(line) for line in lines if line.strip())
    return gpus


def _get_gpu_proc(line, gpu_uuid_to_id_map):
    values = line.split(", ")
    pid = int(values[0])
    process_name = values[1]
    gpu_uuid = values[2]
    gpu_name = values[3]
    used_memory = to_float_or_inf(values[4])
    gpu_id = gpu_uuid_to_id_map.get(gpu_uuid, -1)
    proc = GPUProcess(pid, process_name, gpu_id, gpu_uuid, gpu_name, used_memory)
    return proc


def get_gpu_processes():
    gpu_uuid_to_id_map = {gpu.uuid: gpu.id for gpu in get_gpus()}
    output = subprocess.check_output(shlex.split(NVIDIA_SMI_GET_PROCS))
    lines = output.decode("utf-8").split(os.linesep)
    processes = [
        _get_gpu_proc(line, gpu_uuid_to_id_map) for line in lines if line.strip()
    ]
    return processes


def is_gpu_available(
    gpu, gpu_util_max, mem_util_max, mem_free_min, include_ids, include_uuids
):
    return (
        True
        and (gpu.gpu_util <= gpu_util_max)
        and (gpu.mem_util <= mem_util_max)
        and (gpu.mem_free >= mem_free_min)
        and (gpu.id in include_ids)
        and (gpu.uuid in include_uuids)
    )


def get_available_gpus(
    gpu_util_max=1.0,
    mem_util_max=1.0,
    mem_free_min=0,
    include_ids=None,
    include_uuids=None,
):
    """ Return up to `limit` available cpus """
    # Normalize inputs (include_ids and include_uuis need to be iterables)
    gpus = list(get_gpus())
    include_ids = include_ids or [gpu.id for gpu in gpus]
    include_uuids = include_uuids or [gpu.uuid for gpu in gpus]
    # filter available gpus
    selectors = (
        is_gpu_available(
            gpu, gpu_util_max, mem_util_max, mem_free_min, include_ids, include_uuids
        )
        for gpu in gpus
    )
    available_gpus = it.compress(gpus, selectors)
    return available_gpus


def get_parser():
    main_parser = argparse.ArgumentParser(
        prog="nvsmi", description="A (user-)friendy interface for nvidia-smi"
    )
    main_parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=__version__),
    )

    subparsers = main_parser.add_subparsers(
        help="Choose mode of operation", dest="mode", title="subcommands"
    )  # noqa
    ls_parser = subparsers.add_parser("ls", help="List available GPUs")
    ps_parser = subparsers.add_parser("ps", help="Examine the process of a gpu.")

    # list
    ls_parser.add_argument(
        "--ids", nargs="+", metavar="", help="List of GPU IDs to include"
    )

    ls_parser.add_argument(
        "--uuids", nargs="+", metavar="", help="List of GPU uuids to include"
    )

    ls_parser.add_argument(
        "--limit",
        type=int,
        default=999,
        metavar="",
        help="Limit the number of the GPUs",
    )
    ls_parser.add_argument(
        "--mem-free-min",
        type=float,
        default=0,
        metavar="",
        help="The minimum amount of free memory (in MB)",
    )
    ls_parser.add_argument(
        "--mem-util-max",
        type=int,
        default=100,
        metavar="",
        help="The maximum amount of memory [0, 100]",
    )
    ls_parser.add_argument(
        "--gpu-util-max",
        type=int,
        default=100,
        metavar="",
        help="The maximum amount of load [0, 100]",
    )
    ls_parser.add_argument(
        "--sort",
        default="id",
        choices=["id", "gpu_util", "mem_util"],
        metavar="",
        help="Sort the GPUs using the specified attribute",
    )
    ls_parser.add_argument(
        "--json", action="store_true", help="Show results in JSON format"
    )

    # processes
    ps_parser.add_argument(
        "--ids",
        nargs="+",
        metavar="",
        help="Show only the processes of the GPU matching the provided ids",
    )
    ps_parser.add_argument(
        "--uuids",
        nargs="+",
        metavar="",
        help="Show only the processes of the GPU matching the provided UUIDs",
    )
    ps_parser.add_argument(
        "--json", action="store_true", help="Show results in JSON format"
    )

    return main_parser


def _take(n, iterable):
    "Return first n items of the iterable as a list"
    return it.islice(iterable, n)


def is_nvidia_smi_on_path():
    return shutil.which("nvidia-smi")


def _nvsmi_ls(args):
    gpus = list(
        get_available_gpus(
            gpu_util_max=args.gpu_util_max,
            mem_util_max=args.mem_util_max,
            mem_free_min=args.mem_free_min,
            include_ids=args.ids,
            include_uuids=args.uuids,
        )
    )
    gpus.sort(key=operator.attrgetter(args.sort))
    for gpu in _take(args.limit, gpus):
        output = gpu.to_json() if args.json else gpu
        print(output)


def _nvsmi_ps(args):
    processes = get_gpu_processes()
    if args.ids or args.uuids:
        for proc in processes:
            if proc.gpu_id in args.ids or proc.gpu_uuid in args.uuids:
                output = proc.to_json() if args.json else proc
                print(output)
    else:
        for proc in processes:
            output = proc.to_json() if args.json else proc
            print(output)


def validate_ids_and_uuids(args):
    gpus = list(get_gpus())
    gpu_ids = {gpu.id for gpu in gpus}
    gpu_uuids = {gpu.uuid for gpu in gpus}
    invalid_ids = args.ids.difference(gpu_ids)
    invalid_uuids = args.uuids.difference(gpu_uuids)
    if invalid_ids:
        sys.exit(f"The following GPU ids are not available: {invalid_ids}")
    if invalid_uuids:
        sys.exit(f"The following GPU uuids are not available: {invalid_uuids}")


def _main():
    parser = get_parser()
    args = parser.parse_args()
    # Cast ids and uuids to sets
    if args.mode is None:
        parser.print_help()
        sys.exit()
    else:
        args.ids = set(args.ids) if args.ids else set()
        args.uuids = set(args.uuids) if args.uuids else set()
        validate_ids_and_uuids(args)
        if args.mode == "ls":
            _nvsmi_ls(args)
        elif args.mode == "ps":
            _nvsmi_ps(args)
        else:
            parser.print_help()


if __name__ == "__main__":
    # cli mode
    if not is_nvidia_smi_on_path():
        sys.exit("Error: Couldn't find 'nvidia-smi' in $PATH: %s" % os.environ["PATH"])
    _main()