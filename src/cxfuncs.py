from cxconfig import CXCONFIG
from multiprocessing import shared_memory
import copy
import inspect
import numpy as np
import os
import pickle
import pprint
import psutil
import random
import resource
import signal
import subprocess
import string
import struct
import sys
import time
import torch
    
class CXFUZZ:
    count_class_lookup8 = np.zeros(256,dtype=np.uint8)
    stop_requested = False
    output_dir = ""
    debug_api = ""
    initial_corpus = ""
    map_size = 0
    map_size2 = 0
    full_map_size = 0
    shm = None
    shm2 = None
    trace_bits = None
    trace_bits2 = None
    virgin_dict = None
    virgin_bits = None
    crash_bits = None
    dtypes = None
    interesting_float16_values = None
    interesting_float32_values = None
    interesting_float64_values = None
    interesting_float_values = None
    interesting_int8_values = None
    interesting_int16_values = None
    interesting_int32_values = None
    interesting_int_values = None
    interesting_uint8_values = None
    interesting_uint16_values = None
    interesting_uint32_values = None
    interesting_uint_values = None
    interesting_strings = None
    interesting_kewards = None
    interesting_keyward_values = None
    fuzzing_api_nums = 0
    fuzzing_api_list = []
    queue_hashes = set()
    api_queue_nums = {}
    api_crash_nums = {}
    start_time_s = 0
    api_queue_list = {}
    cur_api_name = ""
    cur_input_file = ""
    queue_minibits_dict = {}
    favored_queue_list = []
    passed_queues_list = []
    queue_fuzzed = {}
    cur_seed_name = ""
    cur_input_args_for_mutate = []
    cur_output_args_of_mutate = []
    cur_input_kwargs_for_mutate = {}
    cur_output_kwargs_of_mutate = {}
    # for api is class
    cur_input_args_for_mutate2 = []
    cur_output_args_of_mutate2 = []
    cur_input_kwargs_for_mutate2 = {}
    cur_output_kwargs_of_mutate2 = {}
    cur_input_args_for_mutate3 = []
    cur_output_args_of_mutate3 = []
    cur_input_kwargs_for_mutate3 = {}
    cur_output_kwargs_of_mutate3 = {}
    total_runs = 0
    passed_runs = 0
    timeout_nums = 0
    memout_nums = 0
    hnc_nums = 0
    asan_crashes = 0
    total_crashes = 0
    cur_map_density = 0
    last_print_time_s = 0
    print_interval = 0
    iterations = 0
    limit_timeout = 0
    limit_memory_m = 0
    api_switch_threshold = 0
    stage_cur = 0
    stage_max = 0
    seed_size_limit = 0
    cur_api_is_class = False
    run_asan_nums = 0
    enable_asan_func_hash_check = None
    enable_asan_trace_bits_hash_check = None
    enable_diff_func_hash_check = None
    enable_diff_trace_bits_hash_check = None
    log_file = None
    shm_clean_threshold = None
    plot_data_file = None

    def __init__(self):
        pass

    def fatal_error(self,error_msg=" "):
        frame = sys._getframe(1)
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        print(f"\033[31m[error]\033[0m:[{filename}:{lineno}]:[{error_msg}]")
        self.exit_fuzz()
        os._exit(0)

    def test(self):
        print("test")
        time.sleep(10)

    def gen_random_with_weithts(self,min,max):
        if not isinstance(min,int) or not isinstance(max,int):
            self.fatal_error()
        if max <= min or min <= 0:
            self.fatal_error()
        nums = list(range(min, max+1))
        weights = [1/n for n in nums]
        results = random.choices(nums, weights=weights, k=1)
        return results[0]

    def register_signal_handlers(self):
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

    def handle_signal(self,signum,frame):
        self.stop_requested = True
        sys.stderr.write("\nTerminating...\n")
        sys.stderr.flush()

    def init_some_thing(self):
        self.init_count_class_lookup8()
        self.fuzzing_api_nums = CXCONFIG.fuzzing_api_nums
        self.map_size = CXCONFIG.map_size
        self.map_size2 = CXCONFIG.map_size2
        self.full_map_size = CXCONFIG.full_map_size
        self.limit_timeout = CXCONFIG.limit_timeout
        self.limit_memory_m = CXCONFIG.limit_memory_m
        self.print_interval = CXCONFIG.print_interval
        self.api_switch_threshold = CXCONFIG.api_switch_threshold
        self.seed_size_limit = CXCONFIG.seed_size_limit
        self.init_trace_bits()
        self.init_virgin_bits()
        self.virgin_dict = {"updated_time_ms":0.0,"updated_count":0}
        self.init_crash_bits()
        self.dtypes = [
        torch.bool,
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.uint16, torch.uint32, torch.uint64,
        torch.float16, torch.float32, torch.float64,
        torch.bfloat16,
        torch.complex32, torch.complex64, torch.complex128,
        torch.qint8, torch.quint8, torch.qint32, torch.quint4x2, torch.quint2x4,
        ]
        self.interesting_float16_values = CXCONFIG.interesting_float16_values
        self.interesting_float32_values = CXCONFIG.interesting_float32_values
        self.interesting_float64_values = CXCONFIG.interesting_float64_values
        self.interesting_float_values = self.interesting_float64_values
        self.interesting_int8_values = CXCONFIG.interesting_int8_values
        self.interesting_int16_values = CXCONFIG.interesting_int16_values
        self.interesting_int32_values = CXCONFIG.interesting_int32_values
        self.interesting_int_values = CXCONFIG.interesting_int_values
        self.interesting_uint8_values = CXCONFIG.interesting_uint8_values
        self.interesting_uint16_values = CXCONFIG.interesting_uint16_values
        self.interesting_uint16_values = CXCONFIG.interesting_uint16_values
        self.interesting_uint_values = CXCONFIG.interesting_uint_values
        self.interesting_strings = CXCONFIG.interesting_strings
        self.start_time_s = int(time.time())
        self.enable_asan_func_hash_check = CXCONFIG.enable_asan_func_hash_check
        self.enable_asan_trace_bits_hash_check = CXCONFIG.enable_asan_trace_bits_hash_check
        self.enable_diff_func_hash_check = CXCONFIG.enable_diff_func_hash_check
        self.enable_diff_trace_bits_hash_check = CXCONFIG.enable_diff_trace_bits_hash_check
        self.shm_clean_threshold = CXCONFIG.shm_clean_threshold

    def init_count_class_lookup8(self):
        self.count_class_lookup8[0] = 0
        self.count_class_lookup8[1] = 1
        self.count_class_lookup8[2] = 2
        self.count_class_lookup8[3] = 4
        self.count_class_lookup8[4:8] = 8
        self.count_class_lookup8[8:16] = 16
        self.count_class_lookup8[16:32] = 32
        self.count_class_lookup8[32:128] = 64
        self.count_class_lookup8[128:256] = 128

    def check_core_pattern(self):
        # emulate afl/afl++. is it necessary?
        try:
            with open("/proc/sys/kernel/core_pattern", "r") as f:
                first_char = f.read(1)
                if first_char == '|':
                    self.fatal_error("echo core >/proc/sys/kernel/core_pattern")
        except:
            self.fatal_error()

    def create_output_folder(self):
        if not self.output_dir:
            self.fatal_error()
        try:
            os.makedirs(self.output_dir, exist_ok=False)
        except:
            self.fatal_error()

    def init_trace_bits(self):
        try:
            shm_name = f"cov_{os.getpid()}_{random.getrandbits(32)}"
            self.shm = shared_memory.SharedMemory(name=shm_name,create=True,size=self.full_map_size)
            self.trace_bits = np.ndarray((self.full_map_size,),dtype=np.uint8,buffer=self.shm.buf)
            self.trace_bits.fill(0)
            shm_name2 = f"cov_{os.getpid()}_{random.getrandbits(32)}"
            self.shm2 = shared_memory.SharedMemory(name=shm_name2,create=True,size=self.map_size2)
            self.trace_bits2 = np.ndarray((self.map_size2,),dtype=np.uint8,buffer=self.shm2.buf)
            self.trace_bits2.fill(0)
        except:
            self.fatal_error()

    def close_trace_bits(self):
        if self.shm is not None:
            try:
                self.shm.close()
                self.shm.unlink()
            except:
                self.fatal_error()
        if self.shm2 is not None:
            try:
                self.shm2.close()
                self.shm2.unlink()
            except:
                self.fatal_error()

    def init_virgin_bits(self):
        try:
            self.virgin_bits = np.full((self.map_size,),fill_value=255,dtype=np.uint8)
        except:
            self.fatal_error()

    def init_crash_bits(self):
        try:
            self.crash_bits = np.full((self.map_size,),fill_value=255,dtype=np.uint8)
        except:
            self.fatal_error()

    def create_api_folders(self):
        if not self.fuzzing_api_list:
            self.fatal_error()
        required_subdirs = ['queue', 'crashes']
        for api_name in self.fuzzing_api_list:
            folder_name = api_name
            api_dir = os.path.join(self.output_dir,folder_name)
            try:
                os.makedirs(api_dir,exist_ok=False)
            except Exception as e:
                self.fatal_error(f"{e}")
            for subdir in required_subdirs:
                subdir_folder = os.path.join(api_dir,subdir)
                try:
                    os.makedirs(subdir_folder,exist_ok=False)
                except:
                    self.fatal_error()

    def create_api_files(self):
        cur_input_file = os.path.join(self.output_dir,'cur_input')
        if os.path.exists(cur_input_file):
            self.fatal_error()
        try:
            cur_input = [[],{}]
            torch.save(cur_input,cur_input_file)
        except:
            self.fatal_error()
        self.cur_input_file = cur_input_file
        log_file = os.path.join(self.output_dir,'log.txt')
        if os.path.exists(log_file):
            self.fatal_error()
        try:
            log_info = "log:\n"
            with open(log_file, 'a') as f:
                f.write(log_info)
        except:
            self.fatal_error()
        self.log_file = log_file
        for api_name in self.fuzzing_api_list:
            folder_name = api_name
            api_dir = os.path.join(self.output_dir,folder_name)
            plot_data_file = os.path.join(api_dir,'plot_data.txt')
            if os.path.exists(plot_data_file):
                self.fatal_error()
            else:
                with open(plot_data_file, 'w') as f:
                    pass
            queue_hashes_file = os.path.join(api_dir,'queue_hashes.pkl')
            if os.path.exists(queue_hashes_file):
                self.fatal_error()
            try:
                queue_hashes = set()
                with open(queue_hashes_file,'wb') as f:
                    pickle.dump(queue_hashes,f)
            except:
                self.fatal_error()
            queue_fuzzed_file = os.path.join(api_dir,'queue_fuzzed.pkl')
            if os.path.exists(queue_fuzzed_file):
                self.fatal_error()
            try:
                with open(queue_fuzzed_file,'wb') as f:
                    pickle.dump({},f)
            except:
                self.fatal_error()
            queue_passed_file = os.path.join(api_dir,'queue_passed.pkl')
            if os.path.exists(queue_passed_file):
                self.fatal_error()
            try:
                with open(queue_passed_file,'wb') as f:
                    pickle.dump([],f)
            except:
                self.fatal_error()
            virgin_bits_file = os.path.join(api_dir,'virgin_bits.npy')
            if os.path.exists(virgin_bits_file):
                self.fatal_error()
            try:
                np.save(virgin_bits_file,self.virgin_bits)
            except:
                self.fatal_error()
            virgin_dict_file = os.path.join(api_dir,'virgin_dict.pkl')
            if os.path.exists(virgin_dict_file):
                self.fatal_error()
            try:
                with open(virgin_dict_file,'wb') as f:
                    pickle.dump({"updated_time_ms":0.0,"updated_count":0},f)
            except:
                self.fatal_error()
            crash_bits_file = os.path.join(api_dir,'crash_bits.npy')
            if os.path.exists(crash_bits_file):
                self.fatal_error()
            try:
                np.save(crash_bits_file,self.crash_bits)
            except:
                self.fatal_error()
    
    def init_kewards_and_values(self):
        self.interesting_keyward_values = [None, True, False]
        self.interesting_keyward_values = self.interesting_keyward_values + self.dtypes
        self.interesting_kewards = []
        pytorch_keywords_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pytorch_keywords.txt')
        try:
            with open(pytorch_keywords_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.interesting_kewards.append(line)
        except:
            self.fatal_error()

    def create_empty_queue(self):
        if not self.fuzzing_api_list:
            self.fatal_error()
        for api_name in self.fuzzing_api_list:
            queue_dir = os.path.join(self.output_dir,api_name,'queue')
            if not os.path.isdir(queue_dir):
                self.fatal_error()
            empty_input = [[],{},[],{}]
            empty_input_id = self.api_queue_nums[api_name]
            cur_time_s = int(time.time())-self.start_time_s
            empty_input_file = f"id:{empty_input_id:06d}-time:{cur_time_s:08d}"
            empty_input_path = os.path.join(queue_dir,empty_input_file)
            if os.path.exists(empty_input_path):
                self.fatal_error()
            try:
                torch.save(empty_input,empty_input_path)
            except:
                self.fatal_error()
            self.api_queue_list[api_name].append(empty_input_file)
            self.api_queue_nums[api_name] += 1
            # minibits
            trace_subset = self.trace_bits[:self.map_size]
            trace_bool = (trace_subset != 0)
            minibits = np.packbits(trace_bool, bitorder='little')
            minibits_file = empty_input_file + "-minibits.npy"
            minibits_path = os.path.join(queue_dir,minibits_file)
            np.save(minibits_path, minibits)

    def load_initial_corpus(self):
        loaded_for_api_nums = 0
        for cur_api in self.fuzzing_api_list:
            self.set_cur_api_to_fuzz(cur_api)
            target = os.path.join(self.initial_corpus, cur_api)
            if os.path.isdir(target):
                loaded_for_api_nums +=1
                files = [os.path.join(target, f) for f in os.listdir(target) if os.path.isfile(os.path.join(target, f))]
                if not files:
                    self.fatal_error(f"no input file in {target}")
                else:
                    for f in files:
                        cur_input = torch.load(f)
                        self.cur_output_args_of_mutate = cur_input[0]
                        self.cur_output_kwargs_of_mutate = cur_input[1]
                        self.cur_output_args_of_mutate2 = cur_input[2]
                        self.cur_output_kwargs_of_mutate2 = cur_input[3]
                        print(f"loading:{f}")
                        status = self.runapi()
                        self.classify_counts()
                        if status < 0:
                            # crash
                            self.fatal_error(f"crash with {f}")
                        else:
                            if status in (2,3):
                                self.fatal_error(f"time/mem out with {f}")
                            elif status ==1:
                                self.fatal_error(f"exception with {f}")
                            elif status == 0:
                                self.passed_runs += 1
                                hnb = self.has_new_bits()
                                if hnb > 0:
                                    self.save_to_queue(status)
                                    self.update_virgin_bits()
                                    self.update_queue_minibits_dict()
                                    self.update_favored_queue()
                                    self.cur_map_density = self.virgin_byte_density()
                                    self.update_plot_data_file()
                            else:
                                self.fatal_error(f"status:{status}")
        print(f"loaded_for_api_nums:{loaded_for_api_nums}")
        time.sleep(3)

    def load_queue_minibits_dict(self):
        self.queue_minibits_dict = {}
        queue_dir = os.path.join(self.output_dir,self.cur_api_name,'queue')
        if not os.path.isdir(queue_dir):
            self.fatal_error()
        for seed in self.api_queue_list[self.cur_api_name]:
            seed_minibits_name = seed + "-minibits.npy"
            seed_minibits_path = os.path.join(queue_dir,seed_minibits_name)
            try:
                loaded_minibits = np.load(seed_minibits_path)
            except:
                self.fatal_error()
            self.queue_minibits_dict[seed] = loaded_minibits

    def update_favored_queue(self):
        self.favored_queue_list = []
        file_sizes = {}
        queue_dir = os.path.join(self.output_dir,self.cur_api_name,'queue')
        if not os.path.isdir(queue_dir):
            self.fatal_error()
        for seed in self.api_queue_list[self.cur_api_name]:
            seed_path = os.path.join(queue_dir,seed)
            total_size = os.path.getsize(seed_path)
            file_sizes[seed] = total_size
        sorted_seeds = sorted(file_sizes.keys(), key=lambda x: file_sizes[x])
        byte_size = self.map_size >> 3
        temp_v = np.full(byte_size,0xFF,dtype=np.uint8)
        for seed in sorted_seeds:
            minibits = self.queue_minibits_dict[seed]
            common_bits = np.bitwise_and(minibits,temp_v)
            if np.any(common_bits):
                temp_v &= np.invert(minibits,dtype=np.uint8)
                self.favored_queue_list.append(seed)

    def load_queue_fuzzed(self):
        self.queue_fuzzed = {}
        api_dir = os.path.join(self.output_dir,self.cur_api_name)
        cur_queue_fuzzed_file = os.path.join(api_dir,'queue_fuzzed.pkl')
        try:
            with open(cur_queue_fuzzed_file, 'rb') as f:
                self.queue_fuzzed = pickle.load(f)
        except:
            self.fatal_error()

    def load_queue_passed(self):
        self.passed_queues_list = []
        api_dir = os.path.join(self.output_dir,self.cur_api_name)
        cur_queue_passed_file = os.path.join(api_dir,'queue_passed.pkl')
        try:
            with open(cur_queue_passed_file, 'rb') as f:
                self.passed_queues_list = pickle.load(f)
        except:
            self.fatal_error()

    def set_cur_api_to_fuzz(self, api_name):
        if api_name not in self.fuzzing_api_list:
            self.fatal_error()
        self.cur_api_name = api_name
        api_dir = os.path.join(self.output_dir,api_name)
        self.plot_data_file = os.path.join(api_dir,'plot_data.txt')
        cur_virgin_bits_file = os.path.join(api_dir,'virgin_bits.npy')
        cur_virgin_dict_file = os.path.join(api_dir,'virgin_dict.pkl')
        cur_crash_bits_file = os.path.join(api_dir,'crash_bits.npy')
        queue_hashes_file = os.path.join(api_dir,'queue_hashes.pkl')
        for file in [cur_virgin_bits_file,cur_virgin_dict_file,cur_crash_bits_file,queue_hashes_file]:
            if not os.path.exists(file):
                self.fatal_error()
        try:
            self.virgin_bits = np.load(cur_virgin_bits_file)
            with open(cur_virgin_dict_file, 'rb') as f:
                self.virgin_dict = pickle.load(f)
            self.crash_bits = np.load(cur_crash_bits_file)
            with open(queue_hashes_file, 'rb') as f:
                self.queue_hashes = pickle.load(f)
            self.load_queue_minibits_dict()
            self.update_favored_queue()
            self.load_queue_fuzzed()
            self.load_queue_passed()
        except:
            self.fatal_error()
        # cur api is class?
        self.cur_api_is_class = False
        api_parts = self.cur_api_name.split('.')
        api_func = torch
        for part in api_parts[1:]:
            api_func = getattr(api_func, part)
        if inspect.isclass(api_func):
            self.cur_api_is_class = True

    def choose_one_seed(self):
        if len(self.api_queue_list[self.cur_api_name]) < 1:
            self.fatal_error()
        passed = [seed for seed in self.passed_queues_list]
        unfuzzed_favored = [seed for seed in self.favored_queue_list 
                            if seed not in self.queue_fuzzed or self.queue_fuzzed[seed] == 0]
        all_unfuzzed = [seed for seed in self.api_queue_list[self.cur_api_name]
                        if seed not in self.queue_fuzzed or self.queue_fuzzed[seed] == 0]
        if unfuzzed_favored:
            choosed_seed = random.choice(unfuzzed_favored)
        elif all_unfuzzed:
            pro = random.random()
            if pro < 0.75 and self.favored_queue_list:
                choosed_seed = random.choice(self.favored_queue_list)
            else:
                choosed_seed = random.choice(all_unfuzzed)
        else:
            pro = random.random()
            if pro < 0.90 and self.favored_queue_list:
                choosed_seed = random.choice(self.favored_queue_list)
            else:
                choosed_seed = random.choice(self.api_queue_list[self.cur_api_name])
        if passed:
            pro = random.random()
            if pro < 0.50:
                choosed_seed = random.choice(self.passed_queues_list)
        self.cur_seed_name = choosed_seed
        queue_dir = os.path.join(self.output_dir,self.cur_api_name,'queue')
        cur_seed_path = os.path.join(queue_dir,self.cur_seed_name)
        try:
            if not os.path.exists(cur_seed_path):
                self.fatal_error()
            cur_seed = torch.load(cur_seed_path)
            if not isinstance(cur_seed, list):
                self.fatal_error()
            if not isinstance(cur_seed[0], list):
                self.fatal_error()
            if not isinstance(cur_seed[1], dict):
                self.fatal_error()
            if not isinstance(cur_seed[2], list):
                self.fatal_error()
            if not isinstance(cur_seed[3], dict):
                self.fatal_error()
            self.cur_input_args_for_mutate = cur_seed[0]
            self.cur_input_kwargs_for_mutate = cur_seed[1]
            if self.cur_api_is_class:
                self.cur_input_args_for_mutate2 = cur_seed[2]
                self.cur_input_kwargs_for_mutate2 = cur_seed[3]
            else:
                self.cur_input_args_for_mutate2 = []
                self.cur_input_kwargs_for_mutate2 = {}
        except:
            self.fatal_error()
        self.cur_output_args_of_mutate = self.cur_input_args_for_mutate.copy()
        self.cur_output_kwargs_of_mutate = self.cur_input_kwargs_for_mutate.copy()
        if self.cur_api_is_class:
            self.cur_output_args_of_mutate2 = self.cur_input_args_for_mutate2.copy()
            self.cur_output_kwargs_of_mutate2 = self.cur_input_kwargs_for_mutate2.copy()
        else:
            self.cur_output_args_of_mutate2 = []
            self.cur_output_kwargs_of_mutate2 = {}
    
    def store_queue_fuzzed(self):
        api_dir = os.path.join(self.output_dir,self.cur_api_name)
        cur_queue_fuzzed_file = os.path.join(api_dir,'queue_fuzzed.pkl')
        with open(cur_queue_fuzzed_file, 'wb') as f:
            pickle.dump(self.queue_fuzzed, f)

    def store_queue_passed(self):
        api_dir = os.path.join(self.output_dir,self.cur_api_name)
        cur_queue_passed_file = os.path.join(api_dir,'queue_passed.pkl')
        with open(cur_queue_passed_file, 'wb') as f:
            pickle.dump(self.passed_queues_list, f)

    def save_cur_api_fuzz_info(self):
        if not self.cur_api_name:
            self.fatal_error()
        api_dir = os.path.join(self.output_dir,self.cur_api_name)
        cur_virgin_bits_file = os.path.join(api_dir, 'virgin_bits.npy')
        cur_virgin_dict_file = os.path.join(api_dir,'virgin_dict.pkl')
        cur_crash_bits_file = os.path.join(api_dir, 'crash_bits.npy')
        cur_queue_hashes_file = os.path.join(api_dir,'queue_hashes.pkl')
        for file in [cur_virgin_bits_file,cur_virgin_dict_file,cur_crash_bits_file,cur_queue_hashes_file]:
            if not os.path.exists(file):
                self.fatal_error()
            os.remove(file)
        try:
            np.save(cur_virgin_bits_file, self.virgin_bits)
            with open(cur_virgin_dict_file,'wb') as f:
                pickle.dump(self.virgin_dict,f)
            np.save(cur_crash_bits_file, self.crash_bits)
            with open(cur_queue_hashes_file,'wb') as f:
                pickle.dump(self.queue_hashes,f)
        except:
            self.fatal_error()
        self.store_queue_fuzzed()
        self.store_queue_passed()
        self.virgin_bits = None
        self.virgin_dict = None
        self.crash_bits = None
        self.cur_api_name = ""

    def numby_count_covered_bytes(self,virgin_bits,map_size):
        if len(virgin_bits) != map_size:
            self.fatal_error()
        return np.sum(virgin_bits != 0xFF)
    
    def virgin_byte_density(self):
        if self.virgin_bits is None:
            self.fatal_error()
        total_bytes = self.map_size
        covered_bytes = self.numby_count_covered_bytes(self.virgin_bits,self.map_size)
        density = np.float64(covered_bytes) / np.float64(total_bytes)
        return round(density, 8)

    def update_queue_fuzzed(self):
        self.queue_fuzzed[self.cur_seed_name] = 1

    def show_stats(self,status=""):
        need_show = 0
        if status:
            need_show = 1
        current_time_s = int(time.time())
        if current_time_s - self.last_print_time_s >= self.print_interval or need_show:
            elapsed_time = current_time_s - self.start_time_s
            self.last_print_time_s = current_time_s
            sys.stdout.write("\033[2J\033[3J\033[H")
            sys.stdout.write(
            # f"\n\n need todo:[] \n\n"
            f"cur status: {status}    \n"
            f"cur api: {self.cur_api_name}    \n"
            f"iterations: {self.iterations} | "
            f"stage cur: {self.stage_cur} | "
            f"stage max: {self.stage_max} | "
            f"trace hash nums: {len(self.queue_hashes)} | "
            f"map density: {self.cur_map_density*100:.2f}% | \n"
            f"time: {elapsed_time}s | "
            f"runok: {self.total_runs} | "
            f"passedok: {self.passed_runs} | "
            f"runasan: {self.run_asan_nums} | "
            f"timeout: {self.timeout_nums} | "
            f"memout: {self.memout_nums} | \n"
            f"queue nums: {self.api_queue_nums[self.cur_api_name]} | "
            f"favored queuen nums: {len(self.favored_queue_list)} | "
            f"passed queuen nums: {len(self.passed_queues_list)} | \n"
            f"crashes: {self.hnc_nums}|{self.asan_crashes}({self.total_crashes})    \n"
            f"func hash updated count: {self.virgin_dict["updated_count"]} | "
            f"func hash key len: {len(self.virgin_dict)} | "
            f"func hash updated time ms: {self.virgin_dict["updated_time_ms"]} | \n"
            )
            sys.stdout.flush()

    def enforce_size_limit(self):
        while True:
            try:
                args_size = len(pickle.dumps(self.cur_output_args_of_mutate))
                if args_size <= self.seed_size_limit:  
                    break
                if self.cur_output_args_of_mutate:
                    count = self.count_all_items(self.cur_output_args_of_mutate)
                    target_id = random.randint(1, count)
                    self.cur_output_args_of_mutate = self.mutate_delete_item(self.cur_output_args_of_mutate, target_id)
                else:
                    self.fatal_error()   
            except:
                self.fatal_error()
        while True:
            try:
                kwargs_size = len(pickle.dumps(self.cur_output_kwargs_of_mutate))
                if kwargs_size <= self.seed_size_limit:  
                    break
                if self.cur_output_kwargs_of_mutate:
                    random_key = random.choice(list(self.cur_output_kwargs_of_mutate.keys()))
                    del self.cur_output_kwargs_of_mutate[random_key]
                else:
                    self.fatal_error()   
            except:
                self.fatal_error()

    def generate_int_value(self, bits=32):
        int_ranges = {
            8: (-128, 127),
            16: (-32768, 32767),
            32: (-2147483648, 2147483647),
            64: (-9223372036854775808, 9223372036854775807),
        }
        if random.random() < 0.5:
            special_values = {
                8: self.interesting_int8_values,
                16: self.interesting_int16_values,
                32: self.interesting_int32_values,
                64: self.interesting_int_values
            }
            return random.choice(special_values[bits])
        else:
            min_val, max_val = int_ranges[bits]
            value = random.randint(min_val, max_val)
            return value

    def generate_uint_value(self, bits=32):
        int_ranges = {
            8: (0, 255),
            16: (0, 65535),
            32: (0, 4294967295),
            64: (0, 18446744073709551615),
        }
        if random.random() < 0.5:
            special_values = {
                8: self.interesting_uint8_values,
                16: self.interesting_uint_values,
                32: self.interesting_uint_values,
                64: self.interesting_uint_values
            }
            return random.choice(special_values[bits])
        else:
            min_val, max_val = int_ranges[bits]
            value = random.randint(min_val, max_val)
            return value

    def generate_str(self):
        max_str_len = 16
        ret = ""
        if random.random() < 0.5:
            ret = random.choice(self.interesting_strings)
        else:
            length = self.gen_random_with_weithts(1,max_str_len)
            ret = ''.join(random.choice(string.printable) for _ in range(length))
        return ret

    def generate_float_value(self, bits=64):
        if bits == 16:
            exponent_range = (-14, 15)
            fraction_mask = 0x3FF
            pack_format = '!H'
            unpack_format = '!e'
        elif bits == 32:
            exponent_range = (-126, 127)
            fraction_mask = 0x7FFFFF
            pack_format = '!I'
            unpack_format = '!f'
        else:
            exponent_range = (-1022, 1023)
            fraction_mask = 0xFFFFFFFFFFFFF
            pack_format = '!Q'
            unpack_format = '!d'
        if random.random() < 0.5:
            special_values = {
                16: self.interesting_float16_values,
                32: self.interesting_float32_values,
                64: self.interesting_float64_values
            }
            return random.choice(special_values[bits])
        else:
            if random.random() < 0.8:
                exponent = random.randint(*exponent_range)
                mantissa = random.uniform(-1.0, 1.0)
                return mantissa * (2.0 ** exponent)
            else:
                sign = random.choice([0, 1])
                if random.random() < 0.1:
                    exponent = 0
                    fraction = random.randint(1, fraction_mask)
                else:
                    exponent = random.randint(1, exponent_range[1] + 127 if bits == 32 else
                                              exponent_range[1] + 1023 if bits == 64 else
                                              exponent_range[1] + 15)
                    fraction = random.randint(0, fraction_mask)
                if bits == 16:
                    packed = (sign << 15) | (exponent << 10) | fraction
                elif bits == 32:
                    packed = (sign << 31) | (exponent << 23) | fraction
                else:
                    packed = (sign << 63) | (exponent << 52) | fraction
                packed_bytes = struct.pack(pack_format, packed)
                [value] = struct.unpack(unpack_format, packed_bytes)
                return value

    def generate_bool_value(self):
        return random.choice([True, False])

    def generate_none(self):
        return None
    
    def generate_empty_bool_tensor(self, shape):
        dtype = torch.bool
        cur_tensor = torch.empty(shape, dtype=dtype)
        return cur_tensor
    
    def generate_bool_tensor(self, shape):
        pro = random.random()
        if pro < 0.1:
            fill_value = random.choice([True, False])
            cur_tensor = torch.full(shape, fill_value, dtype=torch.bool)
            return cur_tensor
        elif pro < 0.2:
            cur_tensor = self.generate_empty_bool_tensor(shape)
            return cur_tensor
        cur_tensor = torch.randint(0, 2, shape, dtype=torch.bool)
        return cur_tensor
    
    def generate_empty_uint_tensor(self, shape, dtype):
        dtypes = [torch.uint8, torch.uint16, torch.uint32, torch.uint64]
        if dtype not in dtypes:
            self.fatal_error()
        cur_tensor = torch.empty(shape, dtype=dtype)
        return cur_tensor
    
    def generate_empty_int_tensor(self, shape, dtype):
        dtypes = [torch.int8, torch.int16, torch.int32, torch.int64]
        if dtype not in dtypes:
            self.fatal_error()
        cur_tensor = torch.empty(shape, dtype=dtype)
        return cur_tensor
    
    def generate_full_uint_tensor(self, shape, dtype):
        dtypes = [torch.uint8, torch.uint16, torch.uint32, torch.uint64]
        if dtype not in dtypes:
            self.fatal_error()
        if dtype == torch.uint8:
            bits = 8
        elif dtype == torch.uint16:
            bits = 16
        elif dtype == torch.uint32:
            bits = 32
        elif dtype == torch.uint64:
            bits = 64
        random_uint = self.generate_uint_value(bits)
        try:
            cur_tensor = torch.full(shape, random_uint, dtype=dtype)
            return cur_tensor
        except:
            self.fatal_error()

    def generate_full_int_tensor(self, shape, dtype):
        dtypes = [torch.int8, torch.int16, torch.int32, torch.int64]
        if dtype not in dtypes:
            self.fatal_error()
        if dtype == torch.int8:
            bits = 8
        elif dtype == torch.int16:
            bits = 16
        elif dtype == torch.int32:
            bits = 32
        elif dtype == torch.int64:
            bits = 64
        random_int = self.generate_int_value(bits)
        try:
            cur_tensor = torch.full(shape, random_int, dtype=dtype)
            return cur_tensor
        except:
            self.fatal_error()

    def generate_uint_tensor(self, shape, dtype):
        dtypes = [torch.uint8, torch.uint16, torch.uint32, torch.uint64]
        if dtype not in dtypes:
            self.fatal_error()
        if random.random() < 0.1:
            generator = random.choice([
                self.generate_full_uint_tensor,
                self.generate_empty_uint_tensor
            ])
            cur_tensor = generator(shape, dtype)
            return cur_tensor
        dtype_config = {
            torch.uint8: {
                'max': 255,
                'min': 0
            },
            torch.uint16: {
                'max': 65535,
                'min': 0
            },
            torch.uint32: {
                'max': 4294967295,
                'min': 0
            },
            torch.uint64: {
                'max': 2**62,
                'min': 0
            }
        }
        config = dtype_config[dtype]
        max_val = config['max']
        min_val = config['min']
        try:
            cur_tensor = torch.randint(min_val, max_val, shape, dtype=dtype)
            return cur_tensor
        except:
            self.fatal_error()

    def generate_int_tensor(self, shape, dtype):
        dtypes = [torch.int8, torch.int16, torch.int32, torch.int64]
        if dtype not in dtypes:
            self.fatal_error()
        if random.random() < 0.1:
            generator = random.choice([
                self.generate_full_int_tensor,
                self.generate_empty_int_tensor
            ])
            cur_tensor = generator(shape, dtype)
            return cur_tensor
        dtype_config = {
            torch.int8: {
                'max': 127,
                'min': -128
            },
            torch.int16: {
                'max': 32767,
                'min': -32768
            },
            torch.int32: {
                'max': 2147483647,
                'min': -2147483648
            },
            torch.int64: {
                'max': 9223372036854775807,
                'min': -9223372036854775808
            }
        }
        config = dtype_config[dtype]
        max_val = config['max']
        min_val = config['min']
        cur_tensor = torch.randint(min_val, max_val, shape, dtype=dtype)
        return cur_tensor

    def generate_full_float_tensor(self, shape, dtype):
        dtypes = [torch.float16, torch.float32, torch.float64, torch.bfloat16]
        if dtype not in dtypes:
            self.fatal_error()
        if dtype == torch.float16:
            bits = 16
        elif dtype == torch.bfloat16:
            bits = 16
        elif dtype == torch.float32:
            bits = 32
        elif dtype == torch.float64:
            bits = 64
        random_float = self.generate_float_value(bits)
        try:
            cur_tensor = torch.full(shape, random_float, dtype=dtype)
            return cur_tensor
        except:
            self.fatal_error()

    def generate_empty_float_tensor(self, shape, dtype):
        dtypes = [torch.float16, torch.float32, torch.float64, torch.bfloat16]
        if dtype not in dtypes:
            self.fatal_error()
        cur_tensor = torch.empty(shape, dtype=dtype)
        return cur_tensor

    def generate_float_tensor(self, shape, dtype):
        dtypes = [torch.float16, torch.float32, torch.float64, torch.bfloat16]
        if dtype not in dtypes:
            self.fatal_error()
        if random.random() < 0.1:
            generator = random.choice([
                self.generate_full_float_tensor,
                self.generate_empty_float_tensor
            ])
            cur_tensor = generator(shape, dtype)
            return cur_tensor
        dtype_config = {
            torch.float16: {
                'max_exp': 8,
                'min_exp': -8
            },
            torch.float32: {
                'max_exp': 16,
                'min_exp': -16
            },
            torch.float64: {
                'max_exp': 32,
                'min_exp': -32
            },
            torch.bfloat16: {
                'max_exp': 8,
                'min_exp': -8
            }
        }
        config = dtype_config[dtype]
        chosen_exp = random.randint(config['min_exp'], config['max_exp'])
        chosen_max = 2.0 ** chosen_exp
        cur_tensor = (torch.rand(shape, dtype=dtype) - 0.5)*2.0*chosen_max
        return cur_tensor

    def generate_empty_complex_tensor(self, shape, dtype=None):
        if dtype:
            if dtype in [torch.complex32, torch.complex64, torch.complex128]:
                cur_tensor = torch.empty(shape, dtype=dtype)
                return cur_tensor
            else:
                self.fatal_error()
        else:
            dtypes = [torch.complex32, torch.complex64, torch.complex128]
            dtype = random.choice(dtypes)
            cur_tensor = torch.empty(shape, dtype=dtype)
            return cur_tensor

    def generate_complex_tensor(self, shape, dtype=None):
        if dtype:
            if dtype == torch.complex32:
                if random.random() < 0.1:
                    cur_tensor = self.generate_empty_complex_tensor(shape, dtype)
                    return cur_tensor
                float_dtype = torch.float16
                tsr = self.generate_float_tensor(shape, float_dtype)
                tsi = self.generate_float_tensor(shape, float_dtype)
                cur_tensor = torch.complex(tsr, tsi)
                return cur_tensor
            elif dtype == torch.complex64:
                if random.random() < 0.1:
                    cur_tensor = self.generate_empty_complex_tensor(shape, dtype)
                    return cur_tensor
                float_dtype = torch.float32
                tsr = self.generate_float_tensor(shape, float_dtype)
                tsi = self.generate_float_tensor(shape, float_dtype)
                cur_tensor = torch.complex(tsr, tsi)
                return cur_tensor
            elif dtype == torch.complex128:
                if random.random() < 0.1:
                    cur_tensor = self.generate_empty_complex_tensor(shape, dtype)
                    return cur_tensor
                float_dtype = torch.float64
                tsr = self.generate_float_tensor(shape, float_dtype)
                tsi = self.generate_float_tensor(shape, float_dtype)
                cur_tensor = torch.complex(tsr, tsi)
                return cur_tensor
            else:
                self.fatal_error()
        else:
            dtypes = [torch.float16, torch.float32, torch.float64]
            if random.random() < 0.1:
                cur_tensor = self.generate_empty_complex_tensor(shape, dtype)
                return cur_tensor
            dtype = random.choice(dtypes)
            tsr = self.generate_float_tensor(shape, dtype)
            tsi = self.generate_float_tensor(shape, dtype)
            cur_tensor = torch.complex(tsr, tsi)
            return cur_tensor
    
    def generate_quantize_tensor(self, shape):
        dtypes = [torch.qint8, torch.quint8, torch.qint32, torch.quint4x2, torch.quint2x4]
        dtype = random.choice(dtypes)
        scale = 0.1 * random.random() + 0.01
        zero_point = random.randint(0, 100)
        float_tensor = self.generate_float_tensor(shape, torch.float32)
        cur_tensor = torch.quantize_per_tensor(
            float_tensor,
            scale=scale,
            zero_point=zero_point,
            dtype=dtype
        )
        return cur_tensor
    
    def generate_sparse_tensor(self, shape):
        dtypes = [
        torch.bool,
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
        torch.float16, torch.float32, torch.float64,
        torch.bfloat16,
        torch.complex32, torch.complex64, torch.complex128,
        ]
        dtype = random.choice(dtypes)
        cur_tensor = None
        if dtype in [torch.bool]:
            cur_tensor = self.generate_bool_tensor(shape)
        elif dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            cur_tensor = self.generate_int_tensor(shape,dtype)
        elif dtype in [torch.uint8]:
            cur_tensor = self.generate_uint_tensor(shape,dtype)
        elif dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
            cur_tensor = self.generate_float_tensor(shape,dtype)
        elif dtype in [torch.complex32, torch.complex64, torch.complex128]:
            cur_tensor = self.generate_complex_tensor(shape,dtype)
        else:
            self.fatal_error()
        sparse_tensor = cur_tensor.to_sparse()
        return sparse_tensor

    def generate_tensor(self):
        max_dims = 6
        max_len = 9
        zero_prob = 0.005
        num_dims = random.randint(1, max_dims)
        shape = []
        for _ in range(num_dims):
            dim_size = random.randint(1, max_len)
            if random.random() < zero_prob:
                dim_size = 0
            shape.append(dim_size)
        shape = tuple(shape)
        try:
            pro = random.random()
            if pro < 0.1:
                cur_tensor = self.generate_bool_tensor(shape)
            elif pro < 0.2:
                cur_tensor = self.generate_complex_tensor(shape)
            # pytorch says "Should we just yank all quantization logic as all relevant parts should have been migrated to AO?"
            # https://github.com/pytorch/pytorch/issues/162801
            # elif pro < 0.3:
            #     cur_tensor = self.generate_quantize_tensor(shape)
            elif pro < 0.3:
                cur_tensor = self.generate_sparse_tensor(shape)
            elif pro < 0.5:
                dtypes = [torch.int8, torch.int16, torch.int32, torch.int64]
                dtype = random.choice(dtypes)
                cur_tensor = self.generate_int_tensor(shape, dtype)
            elif pro < 0.7:
                dtypes = [torch.uint8, torch.uint16,
                          torch.uint32, torch.uint64]
                dtype = random.choice(dtypes)
                cur_tensor = self.generate_uint_tensor(shape, dtype)
            else:
                dtypes = [torch.float16, torch.float32,
                          torch.float64, torch.bfloat16,]
                dtype = random.choice(dtypes)
                cur_tensor = self.generate_float_tensor(shape, dtype)
        except:
            self.fatal_error()
        return cur_tensor
    
    def generate_list(self, max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return [random.choice([
                self.generate_int_value(),
                self.generate_uint_value(),
                self.generate_str(),
                self.generate_float_value(),
                self.generate_bool_value(),
                self.generate_none(),
                self.generate_tensor()
            ])]
        length = self.gen_random_with_weithts(1,5)
        result = []
        for _ in range(length):
            choice = random.choice([
                self.generate_int_value,
                self.generate_uint_value,
                self.generate_str,
                self.generate_float_value,
                self.generate_bool_value,
                self.generate_none,
                self.generate_tensor, self.generate_list, self.generate_tuple])
            if choice in (self.generate_list, self.generate_tuple):
                item = choice(max_depth, current_depth + 1)
            else:
                item = choice()
            result.append(item)
        return result
    
    def generate_tuple(self, max_depth=3, current_depth=0):
        return tuple(self.generate_list(max_depth, current_depth))
    
    def _generate_random_args_item(self):
        generator = random.choice([
            self.generate_int_value,
            self.generate_uint_value,
            self.generate_str,
            self.generate_float_value,
            self.generate_bool_value,
            self.generate_none,
            self.generate_tensor,
            self.generate_list,
            self.generate_tuple
        ])
        if generator in (self.generate_list, self.generate_tuple):
            return generator(max_depth=3)
        else:
            return generator()

    def _generate_random_kwargs_item(self):
        if random.random() < 0.5:
            kwarg_key = random.choice(self.interesting_kewards)
        else:
            kwarg_key = self.generate_str()
        pro = random.random()
        if pro < 0.3:
            kwarg_value = random.choice(self.interesting_keyward_values)
        elif pro < 0.6:
            generator = random.choice([
                self.generate_str,
                self.generate_float_value,
                self.generate_int_value,
                self.generate_uint_value
            ])
            kwarg_value = generator()
        else:
            kwarg_value = self._generate_random_args_item()
        return {kwarg_key: kwarg_value}

    def count_items(self, lst):
        count = 0
        for item in lst:
            if isinstance(item, (list, tuple)):
                count += self.count_items(item)
            else:
                count += 1
        return count

    def count_countainers(self, lst):
        count = 0
        for item in lst:
            if isinstance(item, (list, tuple)):
                count += 1
                count += self.count_countainers(item)
        return count
    
    def count_all_items(self, lst):
        count = 0
        count += self.count_items(lst)
        count += self.count_countainers(lst)
        return count
    
    def count_insert_locations(self, lst):
        count = 0
        count += self.count_items(lst)
        count += 1
        count += 2*self.count_countainers(lst)
        return count
    
    def mutate_delete_item(self, lst, target_id):
        cur_lst = [x for x in lst]
        current_id = 0
        for i in range(len(cur_lst)):
            item = cur_lst[i]
            current_id += 1
            if current_id == target_id:
                cur_lst.pop(i)
                return cur_lst
            if isinstance(item, list):
                count = self.count_all_items(item)
                if current_id < target_id and current_id + count >= target_id:
                    ret = self.mutate_delete_item(item, target_id-current_id)
                    cur_lst[i] = ret
                    return cur_lst
                current_id += count
            if isinstance(item, tuple):
                count = self.count_all_items(item)
                if current_id < target_id and current_id + count >= target_id:
                    item_to_list = list(item)
                    ret = self.mutate_delete_item(
                        item_to_list, target_id-current_id)
                    ret_to_tuple = tuple(ret)
                    cur_lst[i] = ret_to_tuple
                    return cur_lst
                current_id += count
        return cur_lst

    def mutate_add_item(self, lst, target_id, new_item):
        cur_lst = [x for x in lst]
        current_id = 0
        for i in range(len(cur_lst)+1):
            if current_id == target_id:
                cur_lst.insert(i, new_item)
                return cur_lst
            if i < len(cur_lst):
                item = cur_lst[i]
                if isinstance(item, list):
                    count = self.count_insert_locations(item)
                    if current_id <= target_id and current_id + count >= target_id:
                        ret = self.mutate_add_item(
                            item, target_id-current_id-1, new_item)
                        cur_lst[i] = ret
                        return cur_lst
                    current_id += count
                if isinstance(item, tuple):
                    count = self.count_insert_locations(item)
                    if current_id <= target_id and current_id + count >= target_id:
                        item_to_list = list(item)
                        ret = self.mutate_add_item(
                            item_to_list, target_id-current_id-1, new_item)
                        ret_to_tuple = tuple(ret)
                        cur_lst[i] = ret_to_tuple
                        return cur_lst
                    current_id += count
            current_id += 1
        return cur_lst

    def mutate_replace_item(self, lst, target_id, new_item):
        cur_lst = [x for x in lst]
        current_id = 0
        for i in range(len(cur_lst)):
            item = cur_lst[i]
            current_id += 1
            if current_id == target_id:
                cur_lst[i] = new_item
                return cur_lst
            if isinstance(item, list):
                count = self.count_all_items(item)
                if current_id < target_id and current_id + count >= target_id:
                    ret = self.mutate_replace_item(
                        item, target_id-current_id, new_item)
                    cur_lst[i] = ret
                    return cur_lst
                current_id += count
            if isinstance(item, tuple):
                count = self.count_all_items(item)
                if current_id < target_id and current_id + count >= target_id:
                    item_to_list = list(item)
                    ret = self.mutate_replace_item(
                        item_to_list, target_id-current_id, new_item)
                    ret_to_tuple = tuple(ret)
                    cur_lst[i] = ret_to_tuple
                    return cur_lst
                current_id += count
        return cur_lst

    def choose_one_from_items(self, lst, target_id):
        cur_lst = [x for x in lst]
        current_id = 0
        for i in range(len(cur_lst)):
            item = cur_lst[i]
            if not isinstance(item, list) and not isinstance(item, tuple):
                current_id += 1
                if current_id == target_id:
                    return cur_lst[i]
            if isinstance(item, list):
                count = self.count_items(item)
                if current_id < target_id and current_id + count >= target_id:
                    ret = self.choose_one_from_items(
                        item, target_id-current_id)
                    return ret
                current_id += count
            if isinstance(item, tuple):
                count = self.count_items(item)
                if current_id < target_id and current_id + count >= target_id:
                    item_to_list = list(item)
                    ret = self.choose_one_from_items(
                        item_to_list, target_id-current_id)
                    return ret
                current_id += count
        self.fatal_error()

    def choose_one_from_all_items(self, lst, target_id):
        cur_lst = [x for x in lst]
        current_id = 0
        for i in range(len(cur_lst)):
            item = cur_lst[i]
            current_id += 1
            if current_id == target_id:
                return cur_lst[i]
            if isinstance(item, list):
                count = self.count_all_items(item)
                if current_id < target_id and current_id + count >= target_id:
                    ret = self.choose_one_from_all_items(
                        item, target_id-current_id)
                    return ret
                current_id += count
            if isinstance(item, tuple):
                count = self.count_all_items(item)
                if current_id < target_id and current_id + count >= target_id:
                    item_to_list = list(item)
                    ret = self.choose_one_from_all_items(
                        item_to_list, target_id-current_id)
                    return ret
                current_id += count
        self.fatal_error()

    def mutate_replace_item_without_containers(self, lst, target_id, new_item):
        cur_lst = [x for x in lst]
        current_id = 0
        for i in range(len(cur_lst)):
            item = cur_lst[i]
            if not isinstance(item, list) and not isinstance(item, tuple):
                current_id += 1
                if current_id == target_id:
                    cur_lst[i] = new_item
                    return cur_lst
            if isinstance(item, list):
                count = self.count_items(item)
                if current_id < target_id and current_id + count >= target_id:
                    ret = self.mutate_replace_item_without_containers(
                        item, target_id-current_id, new_item)
                    cur_lst[i] = ret
                    return cur_lst
                current_id += count
            if isinstance(item, tuple):
                count = self.count_items(item)
                if current_id < target_id and current_id + count >= target_id:
                    item_to_list = list(item)
                    ret = self.mutate_replace_item_without_containers(
                        item_to_list, target_id-current_id, new_item)
                    ret_to_tuple = tuple(ret)
                    cur_lst[i] = ret_to_tuple
                    return cur_lst
                current_id += count
        self.fatal_error()

    def normal_mutate_args(self):
        count = self.count_items(self.cur_output_args_of_mutate)
        pro = random.random()
        if pro < 0.9 or count < 1:
            num_mutations = self.gen_random_with_weithts(1,10)
        else:
            num_mutations = random.randint(1, count)
        for _ in range(num_mutations):
            if not self.cur_output_args_of_mutate:
                new_item = self._generate_random_args_item()
                self.cur_output_args_of_mutate.append(new_item)
                continue
            operation = random.choice(["delete", "add", "replace", "copy"])
            if operation == "delete":
                count = self.count_all_items(self.cur_output_args_of_mutate)
                target_id = random.randint(1, count)
                self.cur_output_args_of_mutate = self.mutate_delete_item(self.cur_output_args_of_mutate, target_id)
            elif operation == "add":
                new_item = self._generate_random_args_item()
                count = self.count_insert_locations(self.cur_output_args_of_mutate)
                target_id = random.randint(0, count-1)
                self.cur_output_args_of_mutate = self.mutate_add_item(self.cur_output_args_of_mutate, target_id, new_item)
            elif operation == "replace":
                new_item = self._generate_random_args_item()
                count = self.count_all_items(self.cur_output_args_of_mutate)
                target_id = random.randint(1, count)
                self.cur_output_args_of_mutate = self.mutate_replace_item(self.cur_output_args_of_mutate, target_id, new_item)
            elif operation == "copy":
                count = self.count_all_items(self.cur_output_args_of_mutate)
                target_id = random.randint(1, count)
                item = self.choose_one_from_all_items(self.cur_output_args_of_mutate, target_id)
                copyed_item = copy.deepcopy(item)
                target_id = random.randint(1, count)
                self.cur_output_args_of_mutate = self.mutate_replace_item(self.cur_output_args_of_mutate, target_id, copyed_item)
            else:
                self.fatal_error()

    def normal_mutate_kwargs(self):
        count = len(self.cur_output_kwargs_of_mutate)
        pro = random.random()
        if pro < 0.9 or count < 1:
            num_mutations = self.gen_random_with_weithts(1,10)
        else:
            num_mutations = random.randint(1, count)
        for _ in range(num_mutations):
            if not self.cur_output_kwargs_of_mutate:
                new_pair = self._generate_random_kwargs_item()
                self.cur_output_kwargs_of_mutate.update(new_pair)
                continue
            operation = random.choice(["delete", "add", "mut_key", "mut_value"])
            # in dict, replace = delete + add
            if operation == "delete":
                random_key = random.choice(list(self.cur_output_kwargs_of_mutate.keys()))
                del self.cur_output_kwargs_of_mutate[random_key]
            elif operation == "add":
                new_pair = self._generate_random_kwargs_item()
                self.cur_output_kwargs_of_mutate.update(new_pair)
            elif operation == "mut_key":
                old_key = random.choice(list(self.cur_output_kwargs_of_mutate.keys()))
                new_key = self.generate_str()
                if new_key not in self.cur_output_kwargs_of_mutate:
                    old_value = self.cur_output_kwargs_of_mutate[old_key]
                    del self.cur_output_kwargs_of_mutate[old_key]
                    self.cur_output_kwargs_of_mutate[new_key] = old_value
            elif operation == "mut_value":
                key = random.choice(list(self.cur_output_kwargs_of_mutate.keys()))
                value = self.cur_output_kwargs_of_mutate[key]
                if isinstance(value, list):
                    if not value:
                        new_item = self._generate_random_args_item()
                        value.append(new_item)
                    operation2 = random.choice(["delete", "add", "replace", "copy"])
                    if operation2 == "delete":
                        count = self.count_all_items(value)
                        target_id = random.randint(1, count)
                        value = self.mutate_delete_item(value, target_id)
                    elif operation2 == "add":
                        new_item = self._generate_random_args_item()
                        count = self.count_insert_locations(value)
                        target_id = random.randint(0, count-1)
                        value = self.mutate_add_item(value, target_id, new_item)
                    elif operation2 == "replace":
                        new_item = self._generate_random_args_item()
                        count = self.count_all_items(value)
                        target_id = random.randint(1, count)
                        value = self.mutate_replace_item(value, target_id, new_item)
                    elif operation2 == "copy":
                        count = self.count_all_items(value)
                        target_id = random.randint(1, count)
                        item = self.choose_one_from_all_items(value, target_id)
                        copyed_item = copy.deepcopy(item)
                        target_id = random.randint(1, count)
                        value = self.mutate_replace_item(value, target_id, copyed_item)
                    else:
                        self.fatal_error()
                elif isinstance(value, tuple):
                    value_lst = list(value)
                    if not value_lst:
                        new_item = self._generate_random_args_item()
                        value_lst.append(new_item)
                    operation2 = random.choice(["delete", "add", "replace", "copy"])
                    if operation2 == "delete":
                        count = self.count_all_items(value_lst)
                        target_id = random.randint(1, count)
                        value_lst = self.mutate_delete_item(value_lst, target_id)
                    elif operation2 == "add":
                        new_item = self._generate_random_args_item()
                        count = self.count_insert_locations(value_lst)
                        target_id = random.randint(0, count-1)
                        value_lst = self.mutate_add_item(value_lst, target_id, new_item)
                    elif operation2 == "replace":
                        new_item = self._generate_random_args_item()
                        count = self.count_all_items(value_lst)
                        target_id = random.randint(1, count)
                        value_lst = self.mutate_replace_item(value_lst, target_id, new_item)
                    elif operation2 == "copy":
                        count = self.count_all_items(value_lst)
                        target_id = random.randint(1, count)
                        item = self.choose_one_from_all_items(value_lst, target_id)
                        copyed_item = copy.deepcopy(item)
                        target_id = random.randint(1, count)
                        value_lst = self.mutate_replace_item(value_lst, target_id, copyed_item)
                    value_tup = tuple(value_lst)
                    self.cur_output_kwargs_of_mutate[key] = value_tup
                else:
                    new_pair = self._generate_random_kwargs_item()
                    kwarg_key = list(new_pair.keys())[0]
                    kwarg_value = new_pair[kwarg_key]
                    self.cur_output_kwargs_of_mutate[key] = kwarg_value
            else:
                self.fatal_error()

    def MUT_FLIPBIT(self, binary_data, data_length):
        if data_length == 0:
            return binary_data
        byte_pos = random.randint(0, data_length - 1)
        bit_pos = random.randint(0, 7)
        binary_data[byte_pos] ^= (1 << bit_pos)
        return binary_data

    def MUT_INTERESTING8(self, binary_data, data_length):
        if data_length == 0:
            return binary_data
        offset = random.randint(0, data_length - 1)
        interesting_value = random.choice(self.interesting_int8_values)
        binary_data[offset] = interesting_value & 0xFF
        return binary_data

    def MUT_INTERESTING16(self, binary_data, data_length):
        if data_length < 2:
            return binary_data
        offset = random.randint(0, data_length - 2)
        interesting_value = random.choice(self.interesting_int16_values)
        use_big_endian = random.random() < 0.5
        if use_big_endian:
            binary_data[offset] = (interesting_value >> 8) & 0xFF
            binary_data[offset + 1] = interesting_value & 0xFF
        else:
            binary_data[offset] = interesting_value & 0xFF
            binary_data[offset + 1] = (interesting_value >> 8) & 0xFF
        return binary_data

    def MUT_INTERESTING32(self, binary_data, data_length):
        if data_length < 4:
            return binary_data
        offset = random.randint(0, data_length - 4)
        interesting_value = random.choice(self.interesting_int32_values)
        use_big_endian = random.random() < 0.5
        if use_big_endian:
            bytes_to_write = [
                (interesting_value >> 24) & 0xFF,
                (interesting_value >> 16) & 0xFF,
                (interesting_value >> 8) & 0xFF,
                interesting_value & 0xFF
            ]
        else:
            bytes_to_write = [
                interesting_value & 0xFF,
                (interesting_value >> 8) & 0xFF,
                (interesting_value >> 16) & 0xFF,
                (interesting_value >> 24) & 0xFF
            ]
        for i in range(4):
            binary_data[offset + i] = bytes_to_write[i]
        return binary_data

    def MUT_ARITH8_SUB(self, binary_data, data_length):
        if data_length == 0:
            return binary_data
        offset = random.randint(0, data_length - 1)
        delta = random.randint(1, 35)
        binary_data[offset] = (binary_data[offset] - delta) & 0xFF
        return binary_data

    def MUT_ARITH8_ADD(self, binary_data, data_length):
        if data_length == 0:
            return binary_data
        offset = random.randint(0, data_length - 1)
        delta = random.randint(1, 35)
        binary_data[offset] = (binary_data[offset] + delta) & 0xFF
        return binary_data

    def MUT_ARITH16_ADD(self, binary_data, data_length):
        if data_length < 2:
            return binary_data
        offset = random.randint(0, data_length - 2)
        delta = random.randint(1, 35)
        use_big_endian = random.random() < 0.5
        if use_big_endian:
            original_val = (binary_data[offset] << 8) | binary_data[offset + 1]
        else:
            original_val = binary_data[offset] | (binary_data[offset + 1] << 8)
        new_val = (original_val + delta) & 0xFFFF
        if use_big_endian:
            binary_data[offset] = (new_val >> 8) & 0xFF
            binary_data[offset + 1] = new_val & 0xFF
        else:
            binary_data[offset] = new_val & 0xFF
            binary_data[offset + 1] = (new_val >> 8) & 0xFF
        return binary_data

    def MUT_ARITH16_SUB(self, binary_data, data_length):
        if data_length < 2:
            return binary_data
        offset = random.randint(0, data_length - 2)
        delta = random.randint(1, 35)
        use_big_endian = random.random() < 0.5
        if use_big_endian:
            original_val = (binary_data[offset] << 8) | binary_data[offset + 1]
        else:
            original_val = binary_data[offset] | (binary_data[offset + 1] << 8)
        new_val = (original_val - delta) & 0xFFFF
        if use_big_endian:
            binary_data[offset] = (new_val >> 8) & 0xFF
            binary_data[offset + 1] = new_val & 0xFF
        else:
            binary_data[offset] = new_val & 0xFF
            binary_data[offset + 1] = (new_val >> 8) & 0xFF
        return binary_data

    def MUT_ARITH32_ADD(self, binary_data, data_length):
        if data_length < 4:
            return binary_data
        offset = random.randint(0, data_length - 4)
        delta = random.randint(1, 35)
        use_big_endian = random.random() < 0.5
        if use_big_endian:
            original_val = (binary_data[offset] << 24) | \
                (binary_data[offset + 1] << 16) | \
                (binary_data[offset + 2] << 8) | \
                binary_data[offset + 3]
        else:
            original_val = binary_data[offset] | \
                (binary_data[offset + 1] << 8) | \
                (binary_data[offset + 2] << 16) | \
                (binary_data[offset + 3] << 24)
        new_val = (original_val + delta) & 0xFFFFFFFF
        if use_big_endian:
            binary_data[offset] = (new_val >> 24) & 0xFF
            binary_data[offset + 1] = (new_val >> 16) & 0xFF
            binary_data[offset + 2] = (new_val >> 8) & 0xFF
            binary_data[offset + 3] = new_val & 0xFF
        else:
            binary_data[offset] = new_val & 0xFF
            binary_data[offset + 1] = (new_val >> 8) & 0xFF
            binary_data[offset + 2] = (new_val >> 16) & 0xFF
            binary_data[offset + 3] = (new_val >> 24) & 0xFF
        return binary_data

    def MUT_ARITH32_SUB(self, binary_data, data_length):
        if data_length < 4:
            return binary_data
        offset = random.randint(0, data_length - 4)
        delta = random.randint(1, 35)
        use_big_endian = random.random() < 0.5
        if use_big_endian:
            original_val = (binary_data[offset] << 24) | \
                (binary_data[offset + 1] << 16) | \
                (binary_data[offset + 2] << 8) | \
                binary_data[offset + 3]
        else:
            original_val = binary_data[offset] | \
                (binary_data[offset + 1] << 8) | \
                (binary_data[offset + 2] << 16) | \
                (binary_data[offset + 3] << 24)
        new_val = (original_val - delta) & 0xFFFFFFFF
        if use_big_endian:
            binary_data[offset] = (new_val >> 24) & 0xFF
            binary_data[offset + 1] = (new_val >> 16) & 0xFF
            binary_data[offset + 2] = (new_val >> 8) & 0xFF
            binary_data[offset + 3] = new_val & 0xFF
        else:
            binary_data[offset] = new_val & 0xFF
            binary_data[offset + 1] = (new_val >> 8) & 0xFF
            binary_data[offset + 2] = (new_val >> 16) & 0xFF
            binary_data[offset + 3] = (new_val >> 24) & 0xFF
        return binary_data

    def MUT_RAND8(self, binary_data, data_length):
        if data_length == 0:
            return binary_data
        offset = random.randint(0, data_length - 1)
        binary_data[offset] = random.randint(0, 255)
        return binary_data

    def MUT_CLONE_COPY(self, binary_data, data_length):
        if data_length < 2:
            return binary_data
        max_clone_len = min(data_length - 1, 64)
        clone_len = random.randint(1, max_clone_len)
        clone_from = random.randint(0, data_length - clone_len)
        clone_to = random.randint(0, data_length)
        new_data = bytearray()
        new_data.extend(binary_data[:clone_to])
        new_data.extend(binary_data[clone_from:clone_from + clone_len])
        new_data.extend(binary_data[clone_to:])
        return new_data

    def MUT_CLONE_FIXED(self, binary_data, data_length):
        if data_length < 2:
            return binary_data
        clone_len = random.randint(1, min(64, data_length))
        clone_to = random.randint(0, data_length)
        if random.random() < 0.5:
            fixed_byte = random.randint(0, 255)
        else:
            fixed_byte = binary_data[clone_to - 1] if clone_to > 0 else 0
        new_data = bytearray()
        new_data.extend(binary_data[:clone_to])
        new_data.extend(bytes([fixed_byte]) * clone_len)
        new_data.extend(binary_data[clone_to:])
        return new_data

    def MUT_OVERWRITE_COPY(self, binary_data, data_length):
        if data_length < 2:
            return binary_data
        copy_len = random.randint(1, data_length - 1)
        while True:
            copy_from = random.randint(0, data_length - copy_len)
            copy_to = random.randint(0, data_length - copy_len)
            if copy_from != copy_to:
                break
        binary_data[copy_to:copy_to +
                    copy_len] = binary_data[copy_from:copy_from+copy_len]
        return binary_data

    def MUT_OVERWRITE_FIXED(self, binary_data, data_length):
        if data_length < 2:
            return binary_data
        copy_len = random.randint(1, data_length - 1)
        copy_to = random.randint(0, data_length - copy_len)
        if random.random() < 0.5:
            fixed_byte = random.randint(0, 255)
        else:
            fixed_byte = binary_data[copy_to - 1] if copy_to > 0 else 0
        binary_data[copy_to:copy_to+copy_len] = bytes([fixed_byte]) * copy_len
        return binary_data

    def MUT_BYTEADD(self, binary_data, data_length):
        if data_length == 0:
            return binary_data
        offset = random.randint(0, data_length - 1)
        binary_data[offset] = (binary_data[offset] + 1) & 0xFF
        return binary_data

    def MUT_BYTESUB(self, binary_data, data_length):
        if data_length == 0:
            return binary_data
        offset = random.randint(0, data_length - 1)
        binary_data[offset] = (binary_data[offset] - 1) & 0xFF
        return binary_data

    def MUT_FLIP8(self, binary_data, data_length):
        if data_length == 0:
            return binary_data
        offset = random.randint(0, data_length - 1)
        binary_data[offset] ^= 0xFF
        return binary_data

    def MUT_SWITCH(self, binary_data, data_length):
        if data_length < 4:
            return binary_data
        switch_from = random.randint(0, data_length - 1)
        while True:
            switch_to = random.randint(0, data_length - 1)
            if switch_from != switch_to:
                break
        if switch_from < switch_to:
            max_len = min(switch_to - switch_from, data_length - switch_to)
        else:
            max_len = min(switch_from - switch_to, data_length - switch_from)
        switch_len = random.randint(1, max_len)
        temp_buf = bytearray(binary_data[switch_from:switch_from + switch_len])
        binary_data[switch_from:switch_from +
                    switch_len] = binary_data[switch_to:switch_to + switch_len]
        binary_data[switch_to:switch_to + switch_len] = temp_buf
        return binary_data

    def MUT_DEL(self, binary_data, data_length):
        if data_length < 2:
            return binary_data
        del_len = random.randint(1, data_length - 1)
        del_pos = random.randint(0, data_length - del_len)
        new_data = bytearray(binary_data[:del_pos])
        new_data.extend(binary_data[del_pos + del_len:])
        return new_data

    def MUT_SHUFFLE(self, binary_data, data_length):
        if data_length < 4:
            return binary_data
        shuffle_len = random.randint(1, data_length - 1)
        offset = random.randint(0, data_length - shuffle_len)
        for i in range(shuffle_len - 1, 0, -1):
            while True:
                j = random.randint(0, i)
                if i != j:
                    break
            binary_data[offset + i], binary_data[offset +
                                                 j] = binary_data[offset + j], binary_data[offset + i]
        return binary_data

    def MUT_DELONE(self, binary_data, data_length):
        if data_length < 2:
            return binary_data
        del_pos = random.randint(0, data_length - 1)
        new_data = bytearray(binary_data[:del_pos])
        new_data.extend(binary_data[del_pos + 1:])
        return new_data

    def MUT_INSERTONE(self, binary_data, data_length):
        if data_length < 2:
            return binary_data
        insert_pos = random.randint(0, data_length - 1)
        if random.random() < 0.5:
            insert_byte = random.randint(0, 255)
        else:
            insert_byte = binary_data[insert_pos - 1] if insert_pos > 0 else 0
        new_data = bytearray(binary_data[:insert_pos])
        new_data.append(insert_byte)
        new_data.extend(binary_data[insert_pos:])
        return new_data

    def MUT_SPLICE_INSERT_LIST(self, binary_data, data_length):
        queue_dir = os.path.join(self.output_dir,self.cur_api_name,'queue')
        src_file = random.choice(self.api_queue_list[self.cur_api_name])
        src_path = os.path.join(queue_dir,src_file)
        try:
            src_input = torch.load(src_path)
            src_list = src_input[0]
            if not isinstance(src_list, list):
                self.fatal_error()
            src_list = pickle.dumps(src_list)
            insert_pos = random.randint(0, data_length)
            if src_list:
                src_start = random.randint(0, len(src_list)-1)
                splice_len = random.randint(1, len(src_list) - src_start)
                src_end = src_start + splice_len
                binary_data[insert_pos:insert_pos] = src_list[src_start:src_end]
        except Exception as e:
            self.fatal_error()
        return binary_data

    def MUT_SPLICE_OVERWRITE_LIST(self, binary_data, data_length):
        if data_length < 1:
            return binary_data
        queue_dir = os.path.join(self.output_dir,self.cur_api_name,'queue')
        src_file = random.choice(self.api_queue_list[self.cur_api_name])
        src_path = os.path.join(queue_dir,src_file)
        try:
            src_input = torch.load(src_path)
            src_list = src_input[0]
            if not isinstance(src_list, list):
                self.fatal_error()
            src_list = pickle.dumps(src_list)
            if not src_list:
                self.fatal_error()
            target_start = random.randint(0, data_length-1)
            target_end = random.randint(target_start, data_length-1)
            src_start = random.randint(0, len(src_list)-1)
            splice_len = min(target_end - target_start + 1, len(src_list) - src_start)
            src_end = src_start + splice_len
            binary_data[target_start:target_end + 1] = src_list[src_start:src_end]
        except Exception as e:
            self.fatal_error()
        return binary_data

    def MUT_SPLICE_INSERT_DICT(self, binary_data, data_length):
        queue_dir = os.path.join(self.output_dir,self.cur_api_name,'queue')
        src_file = random.choice(self.api_queue_list[self.cur_api_name])
        src_path = os.path.join(queue_dir,src_file)
        try:
            src_input = torch.load(src_path)
            src_dict = src_input[1]
            if not isinstance(src_dict, dict):
                self.fatal_error()
            src_dict = pickle.dumps(src_dict)
            insert_pos = random.randint(0, data_length)
            if src_dict:
                src_start = random.randint(0, len(src_dict)-1)
                splice_len = random.randint(1, len(src_dict) - src_start)
                src_end = src_start + splice_len
                binary_data[insert_pos:insert_pos] = src_dict[src_start:src_end]
        except Exception as e:
            self.fatal_error
        return binary_data

    def MUT_SPLICE_OVERWRITE_DICT(self, binary_data, data_length):
        if data_length < 1:
            return binary_data
        queue_dir = os.path.join(self.output_dir,self.cur_api_name,'queue')
        src_file = random.choice(self.api_queue_list[self.cur_api_name])
        src_path = os.path.join(queue_dir,src_file)
        try:
            src_input = torch.load(src_path)
            src_dict = src_input[1]
            if not isinstance(src_dict, dict):
                self.fatal_error()
            src_dict = pickle.dumps(src_dict)
            if not src_dict:
                self.fatal_error()
            target_start = random.randint(0, data_length-1)
            target_end = random.randint(target_start, data_length-1)
            src_start = random.randint(0, len(src_dict)-1)
            splice_len = min(target_end - target_start + 1, len(src_dict) - src_start)
            src_end = src_start + splice_len
            binary_data[target_start:target_end + 1] = src_dict[src_start:src_end]
        except Exception as e:
            self.fatal_error()
        return binary_data

    def binary_mutate(self, binary_data):
        mutation_strategies = [self.MUT_FLIPBIT, self.MUT_INTERESTING8, self.MUT_INTERESTING16,
                               self.MUT_INTERESTING32, self.MUT_ARITH8_SUB, self.MUT_ARITH8_ADD, self.MUT_ARITH16_ADD, self.MUT_ARITH16_SUB, self.MUT_ARITH32_ADD,
                               self.MUT_ARITH32_SUB, self.MUT_RAND8, self.MUT_CLONE_COPY, self.MUT_CLONE_FIXED, self.MUT_OVERWRITE_COPY, self.MUT_OVERWRITE_FIXED,
                               self.MUT_BYTEADD, self.MUT_BYTESUB, self.MUT_FLIP8, self.MUT_SWITCH, self.MUT_DEL, self.MUT_SHUFFLE, self.MUT_DELONE, self.MUT_INSERTONE,
                               self.MUT_SPLICE_INSERT_LIST, self.MUT_SPLICE_OVERWRITE_LIST]
        num_mutations = random.randint(1, 8)
        for i in range(num_mutations):
            mutator = random.choice(mutation_strategies)
            binary_data = mutator(binary_data, len(binary_data))
        return binary_data

    def binary_mutate_item(self, target_item):
        if isinstance(target_item, bool):
            return random.choice([True, False])
        elif target_item is None:
            return None
        elif torch.is_tensor(target_item):
            is_sparse = False
            if target_item.is_sparse:
                is_sparse = True
                target_item = target_item.to_dense()
            dtype = target_item.dtype
            if dtype in [torch.quint4x2, torch.quint2x4]:
                return target_item.to_sparse() if is_sparse else target_item
            if dtype in [torch.complex32, torch.complex64, torch.complex128]:
                real_part = target_item.real
                imag_part = target_item.imag
                real_numpy = real_part.detach().cpu().numpy()
                imag_numpy = imag_part.detach().cpu().numpy()
                numpy_arr = real_numpy + 1j * imag_numpy
            elif dtype == torch.bfloat16:
                numpy_arr = target_item.float().detach().cpu().numpy()
            elif dtype in [torch.qint8, torch.quint8, torch.qint32]:
                dequantize_target_item = target_item.dequantize()
                numpy_arr = dequantize_target_item.detach().cpu().numpy()
            else:
                numpy_arr = target_item.detach().cpu().numpy()
            if not numpy_arr.flags['C_CONTIGUOUS']:
                numpy_arr = np.ascontiguousarray(numpy_arr)
            orig_bytes = numpy_arr.nbytes
            mem_view = numpy_arr.view(dtype=np.uint8)
            byte_arr = bytearray(mem_view.tobytes())
            mutated_byte_arr = self.binary_mutate(byte_arr)
            mutated_len = len(mutated_byte_arr)
            if mutated_len < orig_bytes:
                padding_length = orig_bytes - len(mutated_byte_arr)
                mutated_byte_arr.extend(random.randbytes(padding_length))
            elif mutated_len > orig_bytes:
                mutated_byte_arr = mutated_byte_arr[:orig_bytes]
            try:
                mutated_np = np.frombuffer(mutated_byte_arr, dtype=np.uint8)
                restored_view = mutated_np.view(numpy_arr.dtype).reshape(numpy_arr.shape)
                if dtype in [torch.qint8, torch.quint8, torch.qint32]:
                    tensor = torch.from_numpy(restored_view).clone().to(dtype=torch.float32)
                    if random.random() < 0.1:
                        if dtype == torch.qint8:
                            zero_point = random.randint(-128, 127)
                        elif dtype == torch.quint8:
                            zero_point = random.randint(0, 255)
                        elif dtype == torch.qint32:
                            if random.random() < 0.1:
                                zero_point = random.randint(-2**31, (2**31)-1)
                            else:
                                zero_point = random.randint(-128, 127)
                        pro = random.random()
                        if pro < 0.25:
                            scale = self.generate_float_value(bits=64)
                        elif pro < 0.5:
                            scale = self.generate_float_value(bits=32)
                        elif pro < 0.75:
                            scale = self.generate_float_value(bits=16)
                        else:
                            orig_scala = target_item.q_scale()
                            scale = self.binary_mutate_item(orig_scala)
                        tensor = torch.quantize_per_tensor(tensor, scale, zero_point, dtype)
                    else:
                        tensor = torch.quantize_per_tensor(tensor, target_item.q_scale(), target_item.q_zero_point(), dtype)
                else:
                    tensor = torch.from_numpy(restored_view).clone().to(dtype=dtype)
                return tensor.to_sparse() if is_sparse else tensor
            except Exception as e:
                self.fatal_error(f"{e}")
        elif isinstance(target_item, str):
            orig_bytes = target_item.encode('utf-8')
            byte_arr = bytearray(orig_bytes)
            orig_len = len(byte_arr)
            mutated_byte_arr = self.binary_mutate(byte_arr)
            if len(mutated_byte_arr) > 128:
                mutated_byte_arr = mutated_byte_arr[:128]
            try:
                mutated_str = mutated_byte_arr.decode('utf-8', errors='ignore')
                if len(mutated_str) > 128:
                    mutated_str = mutated_str[:128]
                return mutated_str
            except UnicodeDecodeError:
                return target_item
        elif isinstance(target_item, int):
            min_val = -(2**127)
            max_val = (2**127) - 1
            original_bytes = target_item.to_bytes(16, byteorder='little', signed=True)
            byte_arr = bytearray(original_bytes)
            original_len = len(byte_arr)
            mutated_byte_arr = self.binary_mutate(byte_arr)
            mutated_len = len(mutated_byte_arr)
            if mutated_len > original_len:
                mutated_byte_arr = mutated_byte_arr[:original_len]
            elif mutated_len < original_len:
                mutated_byte_arr.extend(bytes(original_len - len(mutated_byte_arr)))
            mutated_int = int.from_bytes(
                mutated_byte_arr, byteorder='little', signed=True)
            return max(min_val, min(mutated_int, max_val))
        elif isinstance(target_item, float):
            original_bytes = bytearray(struct.pack('d', target_item))
            original_len = len(original_bytes)
            mutated_byte_arr = self.binary_mutate(original_bytes)
            mutated_len = len(mutated_byte_arr)
            if mutated_len > original_len:
                mutated_byte_arr = mutated_byte_arr[:original_len]
            elif mutated_len < original_len:
                mutated_byte_arr.extend(bytes(original_len - len(mutated_byte_arr)))
            float_bytes = bytes(mutated_byte_arr)
            try:
                ret = struct.unpack('d', float_bytes)[0]
                return ret
            except:
                return target_item
        else:
            self.fatal_error()

    def random_mutate_args(self):
        count = self.count_items(self.cur_output_args_of_mutate)
        if count == 0:
            return False
        if count > 3:
            pro = random.random()
            if pro < 0.5:
                num_mutations = self.gen_random_with_weithts(1, count)
            else:
                num_mutations = random.randint(1, count)
        else: 
            num_mutations = random.randint(1, count)
        for _ in range(num_mutations):
            target_id = random.randint(1, count)
            target_item = self.choose_one_from_items(self.cur_output_args_of_mutate, target_id)
            binary_mutate_item = self.binary_mutate_item(target_item)
            self.cur_output_args_of_mutate = self.mutate_replace_item_without_containers(self.cur_output_args_of_mutate, target_id, binary_mutate_item)

    def random_mutate_kwargs(self):
        if not self.cur_output_kwargs_of_mutate:
            return False
        count2 = len(self.cur_output_kwargs_of_mutate)
        pro = random.random()
        if pro < 0.9 or count2 < 1:
            num_mutations2 = self.gen_random_with_weithts(1,10)
        else:
            num_mutations2 = random.randint(1, count2)
        for _2 in range(num_mutations2):
            key = random.choice(list(self.cur_output_kwargs_of_mutate.keys()))
            value = self.cur_output_kwargs_of_mutate[key]
            if isinstance(value, list):
                count = self.count_items(value)
                if count == 0:
                    return False
                if count > 3:
                    pro = random.random()
                    if pro < 0.5:
                        num_mutations = self.gen_random_with_weithts(1, count)
                    else:
                        num_mutations = random.randint(1, count)
                else: 
                    num_mutations = random.randint(1, count)
                for _ in range(num_mutations):
                    target_id = random.randint(1, count)
                    target_item = self.choose_one_from_items(value, target_id)
                    binary_mutate_item = self.binary_mutate_item(target_item)
                    self.cur_output_kwargs_of_mutate[key] = self.mutate_replace_item_without_containers(value, target_id, binary_mutate_item)
            elif isinstance(value, tuple):
                value_lst = list(value)
                count = self.count_items(value_lst)
                if count == 0:
                    return False
                if count > 3:
                    pro = random.random()
                    if pro < 0.5:
                        num_mutations = self.gen_random_with_weithts(1, count)
                    else:
                        num_mutations = random.randint(1, count)
                else: 
                    num_mutations = random.randint(1, count)
                for _ in range(num_mutations):
                    target_id = random.randint(1, count)
                    target_item = self.choose_one_from_items(value_lst, target_id)
                    binary_mutate_item = self.binary_mutate_item(target_item)
                    value_lst = self.mutate_replace_item_without_containers(value_lst, target_id, binary_mutate_item)
                value_tup = tuple(value_lst)
                self.cur_output_kwargs_of_mutate[key] = value_tup
            elif isinstance(value, bool):
                self.cur_output_kwargs_of_mutate[key] = random.choice([True, False])
            elif value is None:
                pass
            elif torch.is_tensor(value):
                binary_mutate_item = self.binary_mutate_item(value)
                self.cur_output_kwargs_of_mutate[key] = binary_mutate_item
            elif isinstance(value, str):
                binary_mutate_item = self.binary_mutate_item(value)
                self.cur_output_kwargs_of_mutate[key] = binary_mutate_item
            elif value in self.interesting_keyward_values:
                self.cur_output_kwargs_of_mutate[key] = random.choice(self.interesting_keyward_values)
            elif isinstance(value, int):
                binary_mutate_item = self.binary_mutate_item(value)
                self.cur_output_kwargs_of_mutate[key] = binary_mutate_item
            elif isinstance(value, float):
                binary_mutate_item = self.binary_mutate_item(value)
                self.cur_output_kwargs_of_mutate[key] = binary_mutate_item
            else:
                self.fatal_error()

    def mutate(self):
        if self.stop_requested:
            return False
        silence = True
        if silence:
            original_stdout_fd = os.dup(1)
            original_stderr_fd = os.dup(2)
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull_fd, 1)
            os.dup2(devnull_fd, 2)
            os.close(devnull_fd)
        pro = random.random()
        if pro < 0.3:
            self.normal_mutate_args()
        elif pro < 0.7:
            self.random_mutate_args()
        else:
            self.normal_mutate_args()
            self.random_mutate_args()
        if len(self.cur_output_kwargs_of_mutate) == 0:
            if random.random() < 0.9:
                pass
            else:
                pro = random.random()
                if pro < 0.3:
                    self.normal_mutate_kwargs()
                elif pro < 0.7:
                    self.random_mutate_kwargs()
                else:
                    self.normal_mutate_kwargs()
                    self.random_mutate_kwargs()
        else:
            if random.random() < 0.5:
                pass
            else:
                pro = random.random()
                if pro < 0.3:
                    self.normal_mutate_kwargs()
                elif pro < 0.7:
                    self.random_mutate_kwargs()
                else:
                    self.normal_mutate_kwargs()
                    self.random_mutate_kwargs()
        if silence:
            os.dup2(original_stdout_fd, 1)
            os.dup2(original_stderr_fd, 2)
            os.close(original_stdout_fd)
            os.close(original_stderr_fd)
        self.enforce_size_limit()
        if not isinstance(self.cur_output_args_of_mutate, list):
            self.fatal_error()
        if not isinstance(self.cur_output_kwargs_of_mutate, dict):
            self.fatal_error()

    def runapi(self):
        '''
        return status:
        negative: crash
        0/1: run ok. 0 passed, 1 except
        2: timeout
        3: memout
        '''
        if self.stop_requested:
            return 0
        self.total_runs +=1
        self.trace_bits.fill(0)
        self.trace_bits2.fill(0)
        api_parts = self.cur_api_name.split('.')
        api_func = torch
        for part in api_parts[1:]:
            api_func = getattr(api_func, part)
        pid = None
        try:
            pid = os.fork()
            if pid == 0:
                signal.alarm(self.limit_timeout)
                def timeout_handler(signum, frame):
                    os._exit(2)
                signal.signal(signal.SIGALRM, timeout_handler)
                current_vsz_mb = psutil.Process().memory_info().vms / (1024 * 1024)
                dynamic_limit_mb = current_vsz_mb + self.limit_memory_m
                max_bytes = int(dynamic_limit_mb * 1024 * 1024)
                resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
                try:
                    devnull = os.open(os.devnull, os.O_WRONLY)
                    os.dup2(devnull, 1)
                    os.dup2(devnull, 2)
                    os.close(devnull)
                    api_arg_list = self.cur_output_args_of_mutate.copy()
                    api_kwarg_dict = self.cur_output_kwargs_of_mutate.copy()
                    api_arg_list2 = self.cur_output_args_of_mutate2.copy()
                    api_kwarg_dict2 = self.cur_output_kwargs_of_mutate2.copy()
                    c_shm = shared_memory.SharedMemory(name=self.shm.name)
                    trace_bits = np.ndarray((self.full_map_size,), dtype=np.uint8, buffer=c_shm.buf)
                    trace_bits_address = trace_bits.ctypes.data
                    c_shm2 = shared_memory.SharedMemory(name=self.shm2.name)
                    trace_bits2 = np.ndarray((self.map_size2,), dtype=np.uint8, buffer=c_shm2.buf)
                    trace_bits2_address = trace_bits2.ctypes.data
                    os.environ["CXENV1"] = str(int(trace_bits_address))
                    os.environ["CXENV2"] = str(int(trace_bits2_address))
                    r1 = api_func(*api_arg_list, **api_kwarg_dict)
                    if self.cur_api_is_class:
                        r2 = r1(*api_arg_list2, **api_kwarg_dict2)
                    c_shm.close()
                    c_shm2.close()
                    signal.alarm(0)
                    os._exit(0)
                except MemoryError:
                    os.environ["CXENV1"] = str(0)
                    c_shm.close()
                    os.environ["CXENV2"] = str(0)
                    c_shm2.close()
                    os._exit(3)
                except:
                    os.environ["CXENV1"] = str(0)
                    c_shm.close()
                    os.environ["CXENV2"] = str(0)
                    c_shm2.close()
                    os._exit(1)
                finally:
                    os.environ["CXENV1"] = str(0)
                    c_shm.close()
                    os.environ["CXENV2"] = str(0)
                    c_shm2.close()
                    os._exit(1)
            else:
                wait_timeout = self.limit_timeout + 5
                start_time = time.time()
                while True:
                    child_pid, status = os.waitpid(pid, os.WNOHANG)
                    if child_pid != 0:
                        if os.WIFEXITED(status):
                            exit_code = os.WEXITSTATUS(status)
                            if exit_code == 3:
                                result = 3
                            elif exit_code == 2:
                                result = 2
                            elif exit_code == 1:
                                result = 1
                            elif exit_code == 0:
                                result = 0
                            else:
                                self.fatal_error(f"exit_code:{exit_code}")
                        elif os.WIFSIGNALED(status):
                            signal_num = os.WTERMSIG(status)
                            result = -1
                        else:
                            self.fatal_error()
                        break
                    if time.time() - start_time > wait_timeout:
                        try:
                            os.kill(pid, signal.SIGKILL)
                            os.waitpid(pid, 0)
                        except ProcessLookupError:
                            # child process has exited
                            pass
                        except Exception as e:
                            self.fatal_error()
                        result = 2
                        break
                    time.sleep(0.001)
            return result
        except Exception as e:
            self.fatal_error()
        finally:
            os.environ["CXENV1"] = str(0)
            os.environ["CXENV2"] = str(0)
            try:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            except ProcessLookupError:
                # child process has exited
                pass
            except Exception as e:
                self.fatal_error(f"{e}")

    def classify_counts(self):
        np.take(self.count_class_lookup8,self.trace_bits[:self.map_size], out=self.trace_bits[:self.map_size])

    def numby_has_new_crash(self, trace_bits, crash_bits, map_size):
        trace_subset = trace_bits[:map_size]
        mask = (trace_subset & crash_bits) != 0
        if not np.any(mask):
            return 0
        if np.any(crash_bits[mask] == 0xFF):
            crash_bits[mask] &= ~trace_subset[mask]
            return 2
        if np.any(mask):
            crash_bits[mask] &= ~trace_subset[mask]
            return 1
        return 0

    def has_new_crash(self):
        if self.trace_bits is None or self.crash_bits is None:
            self.fatal_error()
        ret = 0
        ret = self.numby_has_new_crash(self.trace_bits,self.crash_bits,self.map_size)
        return ret
    
    def save_to_crash(self):
        crashes_dir = os.path.join(self.output_dir,self.cur_api_name,'crashes')
        if not os.path.isdir(crashes_dir):
            self.fatal_error()
        cur_input = [self.cur_output_args_of_mutate,self.cur_output_kwargs_of_mutate,self.cur_output_args_of_mutate2,self.cur_output_kwargs_of_mutate2]
        try:
            torch.save(cur_input,self.cur_input_file)
            loaded_input = torch.load(self.cur_input_file)
            if not isinstance(loaded_input, list):
                self.fatal_error()
            if not isinstance(loaded_input[0], list):
                self.fatal_error()
            if not isinstance(loaded_input[1], dict):
                self.fatal_error()
            if not isinstance(loaded_input[2], list):
                self.fatal_error()
            if not isinstance(loaded_input[3], dict):
                self.fatal_error()
        except Exception as e:
            with open(self.log_file, "a") as f:
                pprint.pprint(cur_input, stream=f)
            self.fatal_error(f"{e}")
        cur_time_s = int(time.time())-self.start_time_s
        crash_input_id = self.api_crash_nums[self.cur_api_name]
        crash_input_file = f"id:{crash_input_id:06d}-time:{cur_time_s:08d}"
        crash_input_path = os.path.join(crashes_dir,crash_input_file)
        torch.save(cur_input,crash_input_path)
        self.api_crash_nums[self.cur_api_name] += 1

    def numby_update_crash_bits(self, trace_bits, crash_bits, map_size):
        trace_subset = trace_bits[:map_size]
        non_zero_idx = np.nonzero(trace_subset)[0]
        if non_zero_idx.size == 0:
            return False
        tb_subset = trace_subset[non_zero_idx]
        vb_subset = crash_bits[non_zero_idx]
        mask = (tb_subset & vb_subset) != 0
        crash_bits[non_zero_idx[mask]] &= ~tb_subset[mask]

    def update_crash_bits(self):
        if self.trace_bits is None or self.crash_bits is None:
            self.fatal_error()
        self.numby_update_crash_bits(self.trace_bits, self.crash_bits, self.map_size)

    def numby_has_new_bits(self, trace_bits, virgin_bits, map_size):
        trace_subset = trace_bits[:map_size]
        if np.any((virgin_bits == 0xFF) & (trace_subset != 0)):
            return 2
        elif np.any(trace_subset & virgin_bits):
            return 1
        else:
            return 0
    
    def has_new_bits(self):
        if self.trace_bits is None or self.virgin_bits is None:
            self.fatal_error()
        ret = 0
        ret = self.numby_has_new_bits(self.trace_bits, self.virgin_bits, self.map_size)
        return ret
    
    def save_to_queue(self,status):
        queue_dir = os.path.join(self.output_dir,self.cur_api_name,'queue')
        if not os.path.isdir(queue_dir):
            self.fatal_error()
        cur_input = [self.cur_output_args_of_mutate,self.cur_output_kwargs_of_mutate,self.cur_output_args_of_mutate2,self.cur_output_kwargs_of_mutate2]
        try:
            torch.save(cur_input,self.cur_input_file)
            loaded_input = torch.load(self.cur_input_file)
            if not isinstance(loaded_input, list):
                self.fatal_error()
            if not isinstance(loaded_input[0], list):
                self.fatal_error()
            if not isinstance(loaded_input[1], dict):
                self.fatal_error()
            if not isinstance(loaded_input[2], list):
                self.fatal_error()
            if not isinstance(loaded_input[3], dict):
                self.fatal_error()
        except Exception as e:
            with open(self.log_file, "a") as f:
                pprint.pprint(cur_input, stream=f)
            self.fatal_error(f"{e}")
        cur_time_s = int(time.time())-self.start_time_s
        queue_input_id = self.api_queue_nums[self.cur_api_name]
        queue_input_file = f"id:{queue_input_id:06d}-time:{cur_time_s:08d}"
        queue_input_path = os.path.join(queue_dir,queue_input_file)
        torch.save(cur_input,queue_input_path)
        self.api_queue_list[self.cur_api_name].append(queue_input_file)
        self.api_queue_nums[self.cur_api_name] += 1
        # minibits
        trace_subset = self.trace_bits[:self.map_size]
        trace_bool = (trace_subset != 0)
        minibits = np.packbits(trace_bool, bitorder='little')
        minibits_file = queue_input_file + "-minibits.npy"
        minibits_path = os.path.join(queue_dir,minibits_file)
        np.save(minibits_path, minibits)
        if status == 0:
            self.passed_queues_list.append(queue_input_file)
            self.store_queue_passed()

    def update_queue_minibits_dict(self):
        queue_dir = os.path.join(self.output_dir,self.cur_api_name,'queue')
        if not os.path.isdir(queue_dir):
            self.fatal_error()
        file_minibits_name = self.api_queue_list[self.cur_api_name][-1] + "-minibits.npy"
        file_minibits_path = os.path.join(queue_dir, file_minibits_name)
        loaded_minibits = np.load(file_minibits_path)
        self.queue_minibits_dict[self.api_queue_list[self.cur_api_name][-1]] = loaded_minibits

    def numby_update_virgin_bits(self, trace_bits, virgin_bits, map_size):
        trace_subset = trace_bits[:map_size]
        non_zero_idx = np.nonzero(trace_subset)[0]
        if non_zero_idx.size == 0:
            return
        tb_subset = trace_subset[non_zero_idx]
        vb_subset = virgin_bits[non_zero_idx]
        mask = (tb_subset & vb_subset) != 0
        virgin_bits[non_zero_idx[mask]] &= ~tb_subset[mask]

    def update_virgin_bits(self):
        if self.trace_bits is None or self.virgin_bits is None:
            self.fatal_error()
        self.numby_update_virgin_bits(self.trace_bits, self.virgin_bits, self.map_size)

    def save_to_cur_input(self):
        cur_input = [self.cur_output_args_of_mutate,self.cur_output_kwargs_of_mutate,self.cur_output_args_of_mutate2,self.cur_output_kwargs_of_mutate2]
        try:
            torch.save(cur_input,self.cur_input_file)
            loaded_input = torch.load(self.cur_input_file)
            if not isinstance(loaded_input, list):
                self.fatal_error()
            if not isinstance(loaded_input[0], list):
                self.fatal_error()
            if not isinstance(loaded_input[1], dict):
                self.fatal_error()
            if not isinstance(loaded_input[2], list):
                self.fatal_error()
            if not isinstance(loaded_input[3], dict):
                self.fatal_error()
        except Exception as e:
            with open(self.log_file, "a") as f:
                pprint.pprint(cur_input, stream=f)
            self.fatal_error(f"{e}")

    def run_with_asan(self):
        self.run_asan_nums +=1
        crashed = False
        cur_api_is_class = ""
        if self.cur_api_is_class:
            cur_api_is_class = "yes"
        cmd1 = [
            "compute-sanitizer",
            "--print-limit=1",
            "/root/anaconda3/envs/ptn/bin/python",
            "/home/chenxu/fuzz/cxoracle.py",
            "--o",
            self.output_dir,
            "--a",
            self.cur_api_name,
            "--c",
            cur_api_is_class
        ]
        timeout = False
        try:
            result = subprocess.run(
                cmd1,
                timeout=self.limit_timeout,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired:
            timeout = True
        except Exception as e:
            self.fatal_error()
        if timeout:
            pass
        else:
            exit_code = result.returncode
            output = result.stdout
            error = result.stderr
            if "cx error label:" in (output + error):
                self.fatal_error(f"{output};{error}")
            if "cx unsupported:" in (output + error):
                self.fatal_error(f"{error}")
            if exit_code not in [0,1] and \
            "Program hit cudaErrorMemoryAllocation" not in (output + error) and \
            "at __assertfail" not in (output + error):
                crashed = True
                return crashed
        cmd2 = [
            "/root/anaconda3/envs/pta/bin/python",
            "/home/chenxu/fuzz/cxoracle.py",
            "--o",
            self.output_dir,
            "--a",
            self.cur_api_name,
            "--c",
            cur_api_is_class
        ]
        new_env = os.environ.copy()
        new_env["LD_PRELOAD"] = "/usr/lib/llvm-18/lib/clang/18/lib/linux/libclang_rt.asan-x86_64.so"
        new_env["ASAN_OPTIONS"] = "detect_leaks=0"
        timeout = False
        try:
            result = subprocess.run(
                cmd2,
                timeout=self.limit_timeout,
                capture_output=True,
                text=True,
                env=new_env,
            )
        except subprocess.TimeoutExpired:
            timeout = True
        except Exception as e:
            self.fatal_error()
        if timeout:
            pass
        else:
            exit_code = result.returncode
            output = result.stdout
            error = result.stderr
            if "cx error label:" in (output + error):
                self.fatal_error(f"{output};{error}")
            if "cx unsupported:" in (output + error):
                self.fatal_error(f"{error}")
            if exit_code not in [0] and \
            "AddressSanitizer" in (output + error) and \
            "AddressSanitizer: out of memory" not in (output + error) and \
            "AddressSanitizer: allocation-size-too-big" not in (output + error):
                crashed = True
                return crashed
        return crashed

    def run_with_diff_check(self):
        '''
        if isinstance(lhs, torch.Tensor):
            return torch.allclose(
                lhs.cpu(),
                rhs.cpu(),
                rtol=ALLCLOSE_RTOL,
                atol=ALLCLOSE_ATOL,
                equal_nan=True,
            )
        elif isinstance(lhs, int) or isinstance(lhs, float):
            return torch.allclose(
                torch.Tensor([lhs]),
                torch.Tensor([rhs]),
                rtol=ALLCLOSE_RTOL,
                atol=ALLCLOSE_ATOL,
                equal_nan=True,
            )
        elif isinstance(lhs, torch.Size):
            return lhs == rhs
        return True
        '''
        pass

    def cleanup_unused_shm_files(self):
        # when /dev/shm is full, bus error will occur.
        shm_dir = "/dev/shm"
        try:
            files = os.listdir(shm_dir)
        except Exception as e:
            self.fatal_error(f"{e}")
        if not files:
            return
        used_files = set()
        try:
            lsof_output = subprocess.check_output(
                ["lsof", "-F", "n", shm_dir],
                stderr=subprocess.PIPE,
                text=True
            )
            for line in lsof_output.splitlines():
                if line.startswith("n"):
                    filepath = line[1:].strip()
                    filename = os.path.basename(filepath)
                    used_files.add(filename)
        except Exception as e:
            self.fatal_error(f"{e}")
        deleted_files = []
        for filename in files:
            if filename in used_files:
                continue
            filepath = os.path.join(shm_dir, filename)
            if os.path.isfile(filepath):
                try:
                    os.unlink(filepath)
                    deleted_files.append(filename)
                except FileNotFoundError:
                    pass
                except Exception as e:
                    self.fatal_error(f"{e}")

    def update_plot_data_file(self):
        # only for running with one api
        if not os.path.exists(self.plot_data_file):
            self.fatal_error()
        with open(self.plot_data_file, mode='a', encoding='utf-8') as f:
            cur_time_s = int(time.time())-self.start_time_s
            coverage = int(self.cur_map_density * self.map_size)
            content = str(cur_time_s)+","+str(coverage)
            f.write(content + '\n')
            
    def exit_fuzz(self):
        # clean some thing before exit
        self.close_trace_bits()
        print("cxfuzz exited")
