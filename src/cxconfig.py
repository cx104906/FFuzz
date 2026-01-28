class CXCONFIG:
    # should be 2^n
    # should be same in cxconfig.h and cxconfig.py
    map_size = 262144
    full_map_size = map_size + 4
    map_size2 = 524288
    # the number of APIs to be fuzzed in one fuzzer instance
    fuzzing_api_nums = 300
    # child process reusable memory limit (MB)
    limit_memory_m = 4096
    # child process timeout (seconds)
    limit_timeout = 20
    # interesting float values
    interesting_float16_values = [0.0,float('inf'),-float('inf'),float('nan'),1.0,-1.0,0.5,-0.5,2.0,-2.0,
    0.1,-0.1,0.2,-0.2,0.125,-0.125,3.141592653589793,2.718281828459045,1.618033988749895,63.0,-63.0,1024.0,
    -1024.0,65500.0,-65500.0]
    interesting_float32_values = interesting_float16_values + [3.4028234e+38,-3.4028234e+38,1.1754944e-38,
    -1.1754944e-38,1.4012985e-45,-1.4012985e-45,1e+20,-1e+20,1e-20,-1e-20,2147483648.0,-2147483648.0]
    interesting_float64_values = interesting_float32_values + [2.2250738585072014e-308,5e-324,-5e-324,]
    interesting_int8_values = [-128,-64,-32,-16,-8,-4,-2,-1,0,1,2,4,8,16,32,64,127]
    interesting_int16_values = interesting_int8_values + [-32768,-129,128,255,256,512,1000,1024,4096,32767]
    interesting_int32_values = interesting_int16_values + [-2147483648,-100663046,-32769,32768,65535,65536,100663045,2139095040,2147483647]
    interesting_int_values = interesting_int32_values
    interesting_uint8_values = [0,1,2,4,8,16,32,64,128,255]
    interesting_uint16_values = interesting_uint8_values + [256,512,1000,1024,4096,32767,32768,65535]
    interesting_uint32_values = interesting_uint16_values + [65536,100663045,2139095040,2147483647,4294967295]
    interesting_uint_values = interesting_uint8_values + [256,512,1024]
    # interesting strings
    interesting_strings = ["bilinear","circular","constant","cpu","cuda","fro","gpu","max","mean",
    "memory_format","nearest","reflect","relu","replicate","sum","tanh","use_mm_for_euclid_dist_if_necessary",
    "weight","zeros"]
    # how many seconds to print info
    print_interval = 1.0
    # switch the API every x times of fuzz
    api_switch_threshold = 300
    # clean the /dev/shm every x times of fuzz
    shm_clean_threshold = 30
    # seed file max size 5MB
    seed_size_limit = 5*1024*1024
    # enable asan check
    enable_asan_func_hash_check = False
    enable_asan_trace_bits_hash_check = False
    # enable diff check
    enable_diff_func_hash_check = False
    enable_diff_trace_bits_hash_check = False
