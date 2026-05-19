import os
import sys
import importlib
import librosa
import soundfile as sf
import copy
import random
import math
import json
from multiprocessing import Pool, cpu_count

try:
    from scipy.stats import norm
except ImportError:
    print("❌ 错误：'gaussian_mid' 映射器需要 'scipy' 库。")
    print("请运行: pip install scipy")
    sys.exit(1)

# ================================
# 固定路径
# ================================
INPUT_DIR = "dataset_clean"
OUTPUT_DIR = "Voices-in-the-Wild_54c"
NOISES_DIR = "noises"
EFFECTS_PACKAGE = "effects"
USE_GLOBAL_M = True

# ================================
# 配置目录与组合文件的映射关系
# ================================
CONFIG_COMBINATIONS_MAP = {
    "configs": "scene_s",
    "configs_comb": "scene_dt",
    "configs_comb_q": "scene_qp",
}

DEFAULT_CONFIGS_DIR = "configs"

# ================================
# 多进程配置
# ================================
MAX_WORKERS = 32

# ================================
# 变速配置
# ================================
SPEED_MIN = 0.75
SPEED_MAX = 1.5

# ================================
# Gaussian constant
# ================================
GAUSS_MU = 0.5
GAUSS_SIGMA = 1.0 / (2.0 * 1.6448536269514722)


def gaussian_mid_mapper(x):
    p_scaled = x * 0.90 + 0.05
    m = norm.ppf(p_scaled, loc=GAUSS_MU, scale=GAUSS_SIGMA)
    return max(0.0, min(1.0, m))


def linear_mapper(x):
    return x


def sqrt_fwd_mapper(x):
    return math.sqrt(x)


def sqrt_bwd_mapper(x):
    return x ** 2


MAPPERS = {
    "linear": {"func": linear_mapper, "desc": "线性"},
    "sqrt_fwd": {"func": sqrt_fwd_mapper, "desc": "Sqrt forward"},
    "sqrt_bwd": {"func": sqrt_bwd_mapper, "desc": "Sqrt backward"},
    "gaussian_mid": {"func": gaussian_mid_mapper, "desc": "Gaussian 中点聚集"},
}

# ================================
# 允许重复的效果器
# ================================
ALLOW_DUPLICATE_EFFECTS = {"add_noise"}


# ===========================================================
# 变速处理函数
# ===========================================================
def apply_speed_change(audio, sr, speed_factor):
    """
    对音频应用变速处理（不改变音调）
    
    Args:
        audio: 音频数据
        sr: 采样率
        speed_factor: 变速因子 (0.75-1.5)
    
    Returns:
        处理后的音频数据
    """
    return librosa.effects.time_stretch(audio, rate=speed_factor)


# ===========================================================
# 随机参数解析
# ===========================================================
def _resolve_random_params(params_dict, mapper_func, fixed_m=None):
    """
    解析参数。
    如果 fixed_m 不为 None，则强制所有随机参数使用该 m 值（实现全局统一难度）。
    """
    m_logs = {}
    
    for key, value in params_dict.items():
        if isinstance(value, dict) and "random_type" in value:
            # === 核心修改逻辑 ===
            if fixed_m is not None:
                # 使用全局共用的 m
                m = fixed_m
            else:
                # 独立采样
                x = random.random()
                m = mapper_func(x)
            # ===================

            m_logs[key] = round(m, 6)

            rand_type = value.get("random_type")
            is_smaller_ez = value.get("is_smaller_ez", True)

            try:
                if rand_type in ["uniform", "randint"]:
                    min_val = value["min"]
                    max_val = value["max"]
                    val_range = max_val - min_val

                    # m=0 -> Easy, m=1 -> Hard
                    if is_smaller_ez:
                        current_val = min_val + val_range * m
                    else:
                        current_val = max_val - val_range * m

                    if rand_type == "randint":
                        params_dict[key] = int(round(current_val))
                    else:
                        params_dict[key] = current_val

                elif rand_type == "choice":
                    options = value["options"]
                    n = len(options)
                    if n > 0:
                        # 映射 index
                        if is_smaller_ez:
                            target = int(round(m * (n - 1)))
                        else:
                            target = int(round((1 - m) * (n - 1)))
                        target = max(0, min(n - 1, target))
                        params_dict[key] = options[target]

            except Exception as e:
                print(f"⚠ 参数解析失败 {key}: {e}")
                
    return m_logs

def _generate_random_chain(base_effect_chain, mapper_func, fixed_m=None):
    chain = []
    chain_m_info = [] 
    
    for effect_config in base_effect_chain:
        params = copy.deepcopy(effect_config.get("params", {}))
        
        # 将 fixed_m 传递给参数解析函数
        m_logs = _resolve_random_params(params, mapper_func, fixed_m=fixed_m)
        
        chain.append({"name": effect_config["name"], "params": params})
        
        if m_logs:
            chain_m_info.append({
                "effect": effect_config["name"],
                "params_m": m_logs
            })
            
    return chain, chain_m_info


# ===========================================================
# Config 加载
# ===========================================================
def load_all_scene_configs(config_dir):
    """加载所有场景配置，返回 {scene_name: config} 字典"""
    configs = {}
    files = [f for f in os.listdir(config_dir) if f.endswith(".py") and not f.startswith("__")]

    sys.path.insert(0, os.path.abspath(config_dir))
    for filename in files:
        mod = filename[:-3]
        try:
            module = importlib.import_module(mod)
            if hasattr(module, "SCENE_CONFIG"):
                scene_name = module.SCENE_CONFIG.get("scene_name")
                configs[scene_name] = module.SCENE_CONFIG
                print(f"  加载场景配置：{scene_name}")
        except Exception as e:
            print(f"  加载失败 {filename}: {e}")
    sys.path.pop(0)
    return configs


def load_combinations_config(combinations_file):
    """加载场景组合配置"""
    try:
        module = importlib.import_module(combinations_file)
        combinations = getattr(module, "SCENE_COMBINATIONS", [])
        enabled = getattr(module, "ENABLED_COMBINATIONS", [])

        if enabled:
            combinations = [c for c in combinations if c["name"] in enabled]

        return combinations
    except Exception as e:
        print(f"❌ 加载组合配置失败: {e}")
        return []


# ===========================================================
# 效果链合并与去重
# ===========================================================
def merge_effect_chains(scene_configs, scene_names, verbose=True):
    """
    合并多个场景的效果链。
    - 同一场景内的重复效果器：全部保留。
    - 不同场景间的重复效果器：保留靠前场景的，抛弃后面场景的（ALLOW_DUPLICATE_EFFECTS 除外）。
    """
    merged_chain = []
    seen_in_previous_scenes = set()  # 记录之前场景已经出现过的效果器类型

    for scene_name in scene_names:
        if scene_name not in scene_configs:
            if verbose:
                print(f"⚠ 场景 '{scene_name}' 不存在，跳过")
            continue

        effects = scene_configs[scene_name].get("effects", [])
        
        # 记录当前场景中新引入的效果器类型
        # 注意：我们不在遍历当前场景效果器时立即更新 global set，
        # 这样就能允许同一个场景内出现重复项。
        current_scene_new_types = set()

        if verbose:
            print(f"  --> 处理场景: {scene_name}")

        for effect in effects:
            effect_name = effect["name"]

            # 情况 A: 效果器在白名单中，允许跨场景重复（如 add_noise）
            if effect_name in ALLOW_DUPLICATE_EFFECTS:
                merged_chain.append(copy.deepcopy(effect))
                if verbose:
                    print(f"    ✓ [白名单] 效果器 '{effect_name}' 已添加")

            # 情况 B: 效果器在之前的场景中从未出现过
            elif effect_name not in seen_in_previous_scenes:
                merged_chain.append(copy.deepcopy(effect))
                current_scene_new_types.add(effect_name)
                # 注意：即便这里出现了两次相同的效果器，因为 seen_in_previous_scenes 
                # 还没更新，所以它们都会被保留。

            # 情况 C: 效果器已在之前的场景中出现过
            else:
                if verbose:
                    print(f"    ⚠ [跨场景冲突] 效果器 '{effect_name}' 在之前场景已存在，已跳过")

        # 当前场景所有效果器处理完后，更新“已见”集合，影响后续场景
        seen_in_previous_scenes.update(current_scene_new_types)

    return merged_chain


# ===========================================================
# 单个任务处理（供多进程调用）
# ===========================================================
def process_single_task(task):
    """
    处理单个音频增强任务
    """
    input_path = task["input_path"]
    output_path = task["output_path"] 
    base_effect_chain = task["effect_chain"]
    mapper_name = task["mapper_name"]
    meta_info = task.get("meta_info")
    comb_name = task["comb_name"]
    scene_names = task["scene_names"]
    variant_id = task["variant_id"]
    base_name = task["base_name"]
    speed_augment = task.get("speed_augment", False)

    mapper_func = MAPPERS[mapper_name]["func"]

    # 1. 采样全局难度 m
    global_m_value = None
    if USE_GLOBAL_M:
        x_global = random.random()
        global_m_value = max(0.0, min(1.0, mapper_func(x_global)))
        random_chain, chain_m_info = _generate_random_chain(
            base_effect_chain, mapper_func, fixed_m=global_m_value
        )
    else:
        random_chain, chain_m_info = _generate_random_chain(
            base_effect_chain, mapper_func, fixed_m=None
        )

    # 2. 生成变速因子
    speed_factor = random.uniform(SPEED_MIN, SPEED_MAX) if speed_augment else None

    # ============= 【核心修改：一次性确定所有命名，后面不要再改了】 =============
    m_tag = f"_m{global_m_value:.3f}" if global_m_value is not None else ""
    spd_tag = f"_spd{speed_factor:.2f}" if speed_factor is not None else ""
    
    file_name_stem = f"{base_name}_{comb_name}_{mapper_name}_{variant_id}"
    index_str = f"{file_name_stem}{m_tag}{spd_tag}"
    
    dir_name = os.path.dirname(output_path)
    output_path = os.path.join(dir_name, f"{index_str}.wav")
    # ========================================================================

    # 3. 处理音频
    try:
        y, sr = librosa.load(input_path, sr=None)
    except Exception as e:
        return {"success": False, "error": f"读取失败 {input_path}: {e}", "jsonl_entry": None}

    processed = y
    
    # 4. 应用效果链
    for effect in random_chain:
        name = effect["name"]
        params = effect["params"]
        try:
            module = importlib.import_module(f"{EFFECTS_PACKAGE}.{name}")
            fn = getattr(module, "process")
            if name == "add_noise": params["noise_dir"] = NOISES_DIR
            processed = fn(processed, sr, **params)
        except Exception as e:
            return {"success": False, "error": f"效果 {name} 失败: {e}", "jsonl_entry": None}

    # 5. 应用变速
    if speed_factor is not None:
        try:
            processed = apply_speed_change(processed, sr, speed_factor)
        except Exception as e:
            return {"success": False, "error": f"变速失败: {e}", "jsonl_entry": None}

    # 6. 保存文件（这里的 output_path 已经包含了 m 和 spd）
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, processed, sr)
    except Exception as e:
        return {"success": False, "error": f"写入失败 {output_path}: {e}", "jsonl_entry": None}

    # 7. 生成 JSONL 条目
    jsonl_entry = None
    if meta_info:
        subset_name = f"{comb_name}_speed" if speed_augment else comb_name
        
        jsonl_entry = {
            "index": index_str, # 直接使用最上面定义好的 index_str
            "audio_path": os.path.abspath(output_path),
            "question": meta_info.get("question", "Please transcribe the audio content into text."),
            "answer": meta_info.get("answer", meta_info.get("response", "")),
            "subset": subset_name,
            "combination": comb_name,
            "source_scenes": scene_names,
            "aug_params_m": chain_m_info,
            "global_severity": round(global_m_value, 6) if global_m_value is not None else None,
            "speed_factor": round(speed_factor, 2) if speed_factor else None
        }

    return {
        "success": True,
        "error": None,
        "jsonl_entry": jsonl_entry,
        "output_path": output_path
    }
def init_worker():
    """初始化 worker 进程，设置随机种子"""
    import numpy as np
    
    seed = os.getpid() + int.from_bytes(os.urandom(4), 'big')
    random.seed(seed)
    np.random.seed(seed % (2**32))


# ===========================================================
# main
# ===========================================================
def print_usage():
    print("用法：python batch_process_multi.py --num-variants=N --meta=xxx.jsonl [options]")
    print("")
    print("选项：")
    print("  --num-variants=N      每个文件生成的变体数量（必须）")
    print("  --meta=xxx.jsonl      输入元数据文件")
    print("  --mapper=NAME         映射器名称（linear/sqrt_fwd/sqrt_bwd/gaussian_mid）")
    print("  --configs-dir=DIR     配置目录（configs/configs_comb/configs_comb_q）")
    print("  --combinations=FILE   组合配置文件（不含.py，默认根据configs-dir自动匹配）")
    print("  --only=name1,name2    只处理指定的组合名")
    print("  --workers=N           并行进程数（默认32，最大32）")
    print("  --speed-augment       启用随机变速增强（0.75-1.5倍速）")
    print("")
    print("配置目录与组合文件对应关系：")
    for configs_dir, comb_file in CONFIG_COMBINATIONS_MAP.items():
        print(f"  {configs_dir} -> {comb_file}")
    print("")
    print("示例：")
    print("  python batch_process_multi.py --num-variants=3 --meta=data.jsonl --configs-dir=configs")
    print("  python batch_process_multi.py --num-variants=3 --configs-dir=configs_comb --speed-augment")
    print("")
    print("subset 命名规则：")
    print("  不启用变速: subset = 组合名 (如 'noise_reverb')")
    print("  启用变速:   subset = 组合名_speed (如 'noise_reverb_speed')")


def main():
    print("=" * 60)
    print("       多场景组合批量增强（多进程版）")
    print("=" * 60)
    print()

    BASE_DIR = os.path.dirname(INPUT_DIR)

    args = sys.argv[1:]
    num_variants = -1
    mapper_name = "linear"
    meta_path = os.path.join(BASE_DIR, "subset.jsonl")
    configs_dir = DEFAULT_CONFIGS_DIR
    combinations_file = None  # 先设为 None，后面自动匹配
    only_combinations = None
    num_workers = MAX_WORKERS
    speed_augment = False

    for arg in args:
        if arg.startswith("--num-variants="):
            num_variants = int(arg.split("=", 1)[1])
        elif arg.startswith("--mapper="):
            mapper_name = arg.split("=", 1)[1]
        elif arg.startswith("--meta="):
            meta_path = arg.split("=", 1)[1]
        elif arg.startswith("--configs-dir="):
            configs_dir = arg.split("=", 1)[1]
        elif arg.startswith("--combinations="):
            combinations_file = arg.split("=", 1)[1]
        elif arg.startswith("--only="):
            only_combinations = arg.split("=", 1)[1].split(",")
        elif arg.startswith("--workers="):
            num_workers = min(int(arg.split("=", 1)[1]), MAX_WORKERS)
        elif arg == "--speed-augment":
            speed_augment = True
        elif arg in ["-h", "--help"]:
            print_usage()
            return

    # 自动匹配 combinations_file（如果用户没有手动指定）
    if combinations_file is None:
        combinations_file = CONFIG_COMBINATIONS_MAP.get(configs_dir, "scene_s")
        print(f"📎 自动匹配组合配置: {configs_dir} -> {combinations_file}")

    if num_variants <= 0:
        print("❌ 必须指定 --num-variants=N")
        print_usage()
        return

    if mapper_name not in MAPPERS:
        print(f"❌ 未知映射器: {mapper_name}")
        print(f"可用: {list(MAPPERS.keys())}")
        return

    available_cpus = cpu_count()
    num_workers = min(num_workers, available_cpus, MAX_WORKERS)
    print(f"🔧 配置：")
    print(f"   配置目录: {configs_dir}")
    print(f"   组合文件: {combinations_file}")
    print(f"   变体数量: {num_variants}")
    print(f"   映射器: {mapper_name}")
    print(f"   并行进程: {num_workers}")
    print(f"   变速增强: {'✓ 启用 (0.75-1.5x)' if speed_augment else '✗ 禁用'}")
    if speed_augment:
        print(f"   subset后缀: _speed")
        print(f"   输出子目录: <组合名>_speed")
    print()

    # ===========================
    # 加载所有场景配置
    # ===========================
    print("📂 加载场景配置...")
    scene_configs = load_all_scene_configs(configs_dir)
    print(f"  共加载 {len(scene_configs)} 个场景\n")

    # ===========================
    # 加载组合配置
    # ===========================
    print("📂 加载组合配置...")
    combinations = load_combinations_config(combinations_file)

    if only_combinations:
        combinations = [c for c in combinations if c["name"] in only_combinations]

    print(f"  将处理 {len(combinations)} 个组合:")
    for comb in combinations:
        if speed_augment:
            subset_display = f"{comb['name']}_speed"
            output_subdir = f"{comb['name']}_speed"
        else:
            subset_display = comb['name']
            output_subdir = comb['name']
        print(f"    - {comb['name']}: {' + '.join(comb['scenes'])}")
        print(f"      subset: {subset_display}, 输出目录: {output_subdir}/")
    print()

    # ===========================
    # 载入 meta JSONL
    # ===========================
    # meta_dict = {}
    # if meta_path:
    #     print(f"📌 载入 JSONL: {meta_path}")
    #     with open(meta_path, "r", encoding="utf-8") as f:
    #         for line in f:
    #             line = line.strip()
    #             if line:
    #                 obj = json.loads(line)
    #                 meta_dict[obj["index"]] = obj
    #     print(f"  共 {len(meta_dict)} 条记录\n")
    # else:
    #     print("⚠ 未提供 --meta，增强 jsonl 不会生成\n")

    meta_dict = {}
    input_files = []

    if meta_path:
        print(f"📌 载入 JSONL: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)

                # 1️⃣ 记录 meta
                meta_dict[obj["index"]] = obj

                # 2️⃣ 从 meta 中提取音频路径
                audio_path = obj.get("audio_path")
                if audio_path is None:
                    print(f"⚠ meta 缺少 audio_path: {obj.get('index')}")
                    continue

                if not os.path.isfile(audio_path):
                    print(f"⚠ audio_path 不存在: {audio_path}")
                    continue

                input_files.append(audio_path)

        # 去重（防止多个 meta 指向同一音频）
        input_files = sorted(set(input_files))

        print(f"  meta 条目数: {len(meta_dict)}")
        print(f"  实际处理音频数: {len(input_files)}\n")


    # ===========================
    # 创建输出目录
    # ===========================
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ===========================
    # 加载输入文件
    # ===========================
    # input_files = [
    #     os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR)
    #     if f.lower().endswith(".wav")
    # ]
    # input_files = []
    # for root, _, files in os.walk(INPUT_DIR):
    #     for f in files:
    #         if f.lower().endswith(".wav"):
    #             input_files.append(os.path.join(root, f))



    # print(f"📂 找到 {len(input_files)} 个输入文件\n")

    # ===========================
    # 加载输入文件
    # ===========================
    if not meta_path:
        input_files = []
        for root, _, files in os.walk(INPUT_DIR):
            for f in files:
                if f.lower().endswith(".wav"):
                    input_files.append(os.path.join(root, f))

        print(f"📂 从 INPUT_DIR 扫描到 {len(input_files)} 个输入文件\n")
    else:
        print(f"📂 使用 meta 中指定的音频，共 {len(input_files)} 个\n")



    # ===========================
    # 预处理：合并效果链并生成所有任务
    # ===========================
    print("📋 生成任务列表...")
    all_tasks = []

    for comb in combinations:
        comb_name = comb["name"]
        scene_names = comb["scenes"]

        merged_chain = merge_effect_chains(scene_configs, scene_names, verbose=True)

        if not merged_chain:
            print(f"  ⚠ 组合 '{comb_name}' 效果链为空，跳过")
            continue

        print(f"  ✓ 组合 '{comb_name}' 效果链: {[e['name'] for e in merged_chain]}")

        # 根据是否变速决定输出子目录
        if speed_augment:
            output_subdir = f"{comb_name}_speed"
        else:
            output_subdir = comb_name

        for input_path in input_files:
            filename = os.path.basename(input_path)
            base_name = os.path.splitext(filename)[0]
            output_dir = os.path.join(OUTPUT_DIR, output_subdir, base_name)

            meta_info = meta_dict.get(base_name) if meta_dict else None

            for i in range(num_variants):
                variant_id = i + 1
                new_filename = f"{base_name}_{comb_name}_{mapper_name}_{variant_id}.wav"
                output_path = os.path.join(output_dir, new_filename)

                task = {
                    "input_path": input_path,
                    "output_path": output_path,
                    "effect_chain": merged_chain,
                    "mapper_name": mapper_name,
                    "meta_info": meta_info,
                    "comb_name": comb_name,
                    "scene_names": scene_names,
                    "variant_id": variant_id,
                    "base_name": base_name,
                    "speed_augment": speed_augment,
                }
                all_tasks.append(task)

    total_tasks = len(all_tasks)
    print(f"\n📊 总任务数: {total_tasks}")
    print(f"   ({len(combinations)} 组合 × {len(input_files)} 文件 × {num_variants} 变体)\n")

    if total_tasks == 0:
        print("❌ 没有任务需要处理")
        return

    # ===========================
    # 多进程处理
    # ===========================
    print("🚀 开始多进程处理...")
    print("-" * 60)

    success_count = 0
    fail_count = 0
    jsonl_entries = []

    with Pool(processes=num_workers, initializer=init_worker) as pool:
        for i, result in enumerate(pool.imap_unordered(process_single_task, all_tasks), 1):
            if result["success"]:
                success_count += 1
                if result["jsonl_entry"]:
                    jsonl_entries.append(result["jsonl_entry"])
            else:
                fail_count += 1
                print(f"  ❌ {result['error']}")

            if i % 100 == 0 or i == total_tasks:
                progress = i / total_tasks * 100
                print(f"  进度: {i}/{total_tasks} ({progress:.1f}%) - 成功: {success_count}, 失败: {fail_count}")

    print("-" * 60)

    # ===========================
    # 写入 JSONL（追加模式）
    # ===========================
    if jsonl_entries:
        extended_jsonl_path = os.path.join(OUTPUT_DIR, "extended_metadata.jsonl")
        print(f"\n📝 写入 JSONL（追加模式）: {extended_jsonl_path}")

        # 按 index 排序当前批次
        jsonl_entries.sort(key=lambda x: x["index"])

        with open(extended_jsonl_path, "a", encoding="utf-8") as f:
            for entry in jsonl_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"  本次追加 {len(jsonl_entries)} 条记录")
        
        # 统计本次各 subset 数量
        subset_counts = {}
        for entry in jsonl_entries:
            subset = entry["subset"]
            subset_counts[subset] = subset_counts.get(subset, 0) + 1
        print("  本次各 subset 统计:")
        for subset, count in sorted(subset_counts.items()):
            print(f"    - {subset}: {count} 条")

    # ===========================
    # 最终统计
    # ===========================
    print()
    print("=" * 60)
    print("🎉 处理完成！")
    print("=" * 60)
    print(f"  配置目录:   {configs_dir}")
    print(f"  组合文件:   {combinations_file}")
    print(f"  总任务数:   {total_tasks}")
    print(f"  成功:       {success_count}")
    print(f"  失败:       {fail_count}")
    print(f"  成功率:     {success_count/total_tasks*100:.2f}%")
    print(f"  输出目录:   {OUTPUT_DIR}")
    if speed_augment:
        print(f"  变速增强:   已启用 ({SPEED_MIN}-{SPEED_MAX}x)")
        print(f"  subset命名: <组合名>_speed")
    if jsonl_entries:
        print(f"  JSONL文件:  {os.path.join(OUTPUT_DIR, 'extended_metadata.jsonl')} (追加模式)")
    print("=" * 60)


if __name__ == "__main__":
    main()