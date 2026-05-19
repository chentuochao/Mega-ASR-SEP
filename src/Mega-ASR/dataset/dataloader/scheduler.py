#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import argparse
import importlib
import ast
from multiprocessing import Pool, cpu_count

# ====== 复用 executor ======
import batch_process_comb_m as execu


# ==============================
# 自定义异常
# ==============================
class ConfigLoadError(RuntimeError):
    """配置目录或配置文件读取失败。"""


class MissingConfigError(RuntimeError):
    """组合需要的 scene config 不存在。"""


class MissingEffectError(RuntimeError):
    """组合对应的 effect_chain 为空或无法构造。"""


# ==============================
# 配置映射表
# ==============================
MODULE_CONFIG_MAP = {
    "scene_s": "configs",
    "scene_dt": "configs_comb",
    "scene_qp": "configs_comb_q",
}

SINGLE_SELECTED = {
    "barrier",
    "crosstalk",
    "distortion",
    "far_field",
    "noise",
    "strong_echo",
    "stutter",
}

IMPORTANT_DOUBLE_SELECTED = {
    "far_field_crosstalk",
    "far_field_noise",
    "far_field_stutter",
    "noise_stutter",
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="输入原始 meta jsonl")
    ap.add_argument("--single-ratio", type=float, default=0.6)
    ap.add_argument("--important-ratio", type=float, default=0.24)
    ap.add_argument("--sampling-times", type=int, default=1)
    ap.add_argument("--num-variants", type=int, default=1)
    ap.add_argument("--mapper", default="linear")
    ap.add_argument("--speed-augment", action="store_true")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--output-jsonl", default="extended_metadata.jsonl")

    # ======= DEBUG 参数 =======
    ap.add_argument("--debug", action="store_true", help="开启调试模式，打印详细映射关系")
    ap.add_argument("--debug-limit", type=int, default=5, help="debug 模式下每个组合打印多少条详细任务信息")

    return ap.parse_args()


def load_combination_modules(debug=False):
    all_combs = {}
    enabled_total = set()

    if debug:
        print("\n🔍 [DEBUG] 开始扫描组合定义模块...")

    for mod_name, config_dir in MODULE_CONFIG_MAP.items():
        try:
            mod = importlib.import_module(mod_name)
        except ImportError as e:
            raise ConfigLoadError(
                f"无法加载组合定义模块: {mod_name}.py\n"
                f"请检查该模块是否存在，以及当前工作目录是否正确。\n"
                f"原始错误: {repr(e)}"
            ) from e

        combs_in_mod = getattr(mod, "SCENE_COMBINATIONS", None)
        if combs_in_mod is None:
            raise ConfigLoadError(
                f"模块 {mod_name}.py 中缺少 SCENE_COMBINATIONS。"
            )

        enabled = getattr(mod, "ENABLED_COMBINATIONS", None)
        if enabled is None:
            raise ConfigLoadError(
                f"模块 {mod_name}.py 中缺少 ENABLED_COMBINATIONS。"
            )

        for c in combs_in_mod:
            if not isinstance(c, dict):
                raise ConfigLoadError(
                    f"模块 {mod_name}.py 中存在非法组合定义，不是 dict: {repr(c)}"
                )

            if "name" not in c:
                raise ConfigLoadError(
                    f"模块 {mod_name}.py 中存在组合缺少 name 字段: {repr(c)}"
                )

            if "scenes" not in c:
                raise ConfigLoadError(
                    f"模块 {mod_name}.py 中组合 {c.get('name')} 缺少 scenes 字段。"
                )

            name = c["name"]

            if name in all_combs:
                raise ConfigLoadError(f"重复组合名: {name}")

            # 避免直接污染原模块里的 dict
            c = dict(c)
            c["config_source"] = config_dir
            c["defined_in_module"] = mod_name
            all_combs[name] = c

            if debug:
                print(
                    f"  [Found] 组合 '{name}' -> "
                    f"定义于: {mod_name}.py -> "
                    f"匹配路径: {config_dir}/{name}.py 或其子场景配置"
                )

        enabled_total |= set(enabled)

    missing_enabled = enabled_total - set(all_combs.keys())
    if missing_enabled:
        raise ConfigLoadError(
            "ENABLED_COMBINATIONS 中存在未定义的组合名:\n"
            f"{sorted(missing_enabled)}"
        )

    return all_combs, enabled_total


def parse_config_py_file(file_path):
    """
    读取单个 .py 配置文件。

    支持两种形式：
    1. CONFIG = {...}
    2. 文件内容本身就是一个 dict 字面量
    3. 兜底：exec 后取第一个 dict 类型变量
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        raise ConfigLoadError(
            f"无法读取配置文件: {file_path}\n"
            f"原始错误: {repr(e)}"
        ) from e

    # 第一优先级：解析 CONFIG = {...}
    try:
        tree = ast.parse(content, filename=file_path)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "CONFIG":
                        value = ast.literal_eval(node.value)
                        if not isinstance(value, dict):
                            raise ConfigLoadError(
                                f"配置文件 {file_path} 中 CONFIG 不是 dict。"
                            )
                        return value
    except ConfigLoadError:
        raise
    except Exception:
        pass

    # 第二优先级：处理 “变量名 = {...}” 或直接 "{...}"
    try:
        stripped = content.strip()

        if "=" in stripped:
            dict_str = stripped.split("=", 1)[1].strip()
        else:
            dict_str = stripped

        value = ast.literal_eval(dict_str)
        if isinstance(value, dict):
            return value
    except Exception:
        pass

    # 第三优先级：exec 兜底
    try:
        local_vars = {}
        exec(content, {}, local_vars)

        # 优先取 CONFIG
        if "CONFIG" in local_vars:
            value = local_vars["CONFIG"]
            if not isinstance(value, dict):
                raise ConfigLoadError(
                    f"配置文件 {file_path} 中 CONFIG 不是 dict。"
                )
            return value

        # 否则取第一个 dict
        dict_vars = {
            k: v for k, v in local_vars.items()
            if isinstance(v, dict) and not k.startswith("__")
        }

        if len(dict_vars) == 1:
            return next(iter(dict_vars.values()))

        if len(dict_vars) > 1:
            raise ConfigLoadError(
                f"配置文件 {file_path} 中存在多个 dict 变量，无法判断哪个是配置。\n"
                f"候选变量: {list(dict_vars.keys())}\n"
                f"建议统一写成 CONFIG = {{...}}"
            )

        raise ConfigLoadError(
            f"配置文件 {file_path} 中没有找到任何 dict 配置。"
        )

    except ConfigLoadError:
        raise
    except Exception as e:
        raise ConfigLoadError(
            f"无法解析配置文件: {file_path}\n"
            f"请检查是否为 CONFIG = {{...}} 格式，或者文件内容是否为合法 Python dict。\n"
            f"原始错误: {repr(e)}"
        ) from e


def safe_load_configs_from_path(dir_path):
    """
    从指定目录加载所有 .py 场景配置。

    返回:
        {
            "noise": {...},
            "far_field": {...},
            ...
        }

    注意：
    - 目录不存在直接 raise
    - 文件解析失败直接 raise
    - 没有读到任何 .py 配置也直接 raise
    """
    configs = {}
    abs_dir = os.path.abspath(dir_path)

    if not os.path.exists(abs_dir):
        raise ConfigLoadError(
            f"配置目录不存在: {abs_dir}\n"
            f"请检查 MODULE_CONFIG_MAP 中的路径是否正确。"
        )

    if not os.path.isdir(abs_dir):
        raise ConfigLoadError(
            f"配置路径不是目录: {abs_dir}"
        )

    py_files = [
        fn for fn in os.listdir(abs_dir)
        if fn.endswith(".py") and not fn.startswith("__")
    ]

    if not py_files:
        raise ConfigLoadError(
            f"配置目录中没有找到任何 .py 配置文件: {abs_dir}"
        )

    for filename in sorted(py_files):
        scene_name = filename[:-3]
        file_path = os.path.join(abs_dir, filename)
        configs[scene_name] = parse_config_py_file(file_path)

    if not configs:
        raise ConfigLoadError(
            f"配置目录没有成功加载任何配置: {abs_dir}"
        )

    return configs


def validate_scene_configs_for_comb(comb, current_scene_configs):
    """
    检查某个组合需要的所有 scene 是否都能在当前配置目录中找到。
    如果找不到，直接 raise MissingConfigError。
    """
    c_name = comb["name"]
    config_source = comb["config_source"]
    scenes = comb.get("scenes", [])

    if not scenes:
        raise MissingConfigError(
            f"组合 {c_name} 的 scenes 为空。\n"
            f"来源模块: {comb.get('defined_in_module')}.py\n"
            f"配置目录: {config_source}/"
        )

    missing = [sn for sn in scenes if sn not in current_scene_configs]

    if missing:
        available = sorted(current_scene_configs.keys())
        raise MissingConfigError(
            f"组合 {c_name} 读取不到所需 scene config。\n"
            f"来源模块: {comb.get('defined_in_module')}.py\n"
            f"配置目录: {config_source}/\n"
            f"需要的 scenes: {scenes}\n"
            f"缺失的 scenes: {missing}\n"
            f"当前目录已加载的 configs: {available}\n"
            f"请检查是否存在对应文件，例如: "
            f"{', '.join([os.path.join(config_source, sn + '.py') for sn in missing])}"
        )

    for sn in scenes:
        cfg = current_scene_configs.get(sn)

        if cfg is None:
            raise MissingConfigError(
                f"scene {sn} 的配置为 None。\n"
                f"组合: {c_name}\n"
                f"配置目录: {config_source}/"
            )

        if not isinstance(cfg, dict):
            raise MissingConfigError(
                f"scene {sn} 的配置不是 dict。\n"
                f"组合: {c_name}\n"
                f"配置目录: {config_source}/\n"
                f"实际类型: {type(cfg)}"
            )


def build_effect_chain_or_raise(current_scene_configs, comb):
    """
    构造 effect_chain。

    如果 merge_effect_chains 返回空，说明 effect/config 没有正确读到，
    直接 raise MissingEffectError。
    """
    c_name = comb["name"]
    config_source = comb["config_source"]
    scenes = comb["scenes"]

    validate_scene_configs_for_comb(comb, current_scene_configs)

    try:
        merged_chain = execu.merge_effect_chains(
            current_scene_configs,
            scenes,
            verbose=False,
        )
    except Exception as e:
        raise MissingEffectError(
            f"组合 {c_name} 构造 effect_chain 时失败。\n"
            f"配置目录: {config_source}/\n"
            f"scenes: {scenes}\n"
            f"原始错误: {repr(e)}"
        ) from e

    if not merged_chain:
        scene_debug = {
            sn: current_scene_configs.get(sn)
            for sn in scenes
        }

        raise MissingEffectError(
            f"组合 {c_name} 读取不到有效 effect_chain。\n"
            f"这通常说明 scene config 中 effect/effects/effect_chain 字段缺失，"
            f"或者 merge_effect_chains 没有从配置中合并出任何效果。\n"
            f"来源模块: {comb.get('defined_in_module')}.py\n"
            f"配置目录: {config_source}/\n"
            f"scenes: {scenes}\n"
            f"对应 scene config 内容如下:\n"
            f"{json.dumps(scene_debug, ensure_ascii=False, indent=2)}"
        )

    return merged_chain


def main():
    args = parse_args()
    random.seed(42)

    # ---------- 加载 meta ----------
    meta_dict = {}
    input_files = []

    with open(args.meta, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)
            meta_dict[obj["index"]] = obj

            audio_path = obj.get("audio_path")
            if audio_path and os.path.isfile(audio_path):
                input_files.append(audio_path)

    input_files = sorted(set(input_files))
    print(f"📥 载入音频数: {len(input_files)}")

    if not input_files:
        raise RuntimeError(
            f"没有从 meta 中读到任何有效音频文件。\n"
            f"请检查 --meta 文件中的 audio_path 是否存在。"
        )

    # ---------- 加载组合 ----------
    all_combs, enabled_total = load_combination_modules(debug=args.debug)

    # ---------- 安全过滤 ----------
    single_selected = SINGLE_SELECTED & enabled_total
    important_selected = IMPORTANT_DOUBLE_SELECTED & enabled_total
    other_selected = enabled_total - single_selected - important_selected

    single_combs = [all_combs[n] for n in sorted(single_selected)]
    important_combs = [all_combs[n] for n in sorted(important_selected)]
    other_combs = [all_combs[n] for n in sorted(other_selected)]

    if not single_combs and not important_combs and not other_combs:
        raise RuntimeError(
            "没有任何可用组合。\n"
            "请检查 ENABLED_COMBINATIONS 是否为空，或者组合名是否与 SCENE_COMBINATIONS 对应。"
        )

    # ---------- 分配逻辑 ----------
    expanded_inputs = []
    for p in input_files:
        for _ in range(args.sampling_times):
            expanded_inputs.append(p)

    random.shuffle(expanded_inputs)

    total = len(expanded_inputs)
    n_single = int(total * args.single_ratio)
    n_important = int(total * args.important_ratio)

    single_slots = expanded_inputs[:n_single]
    important_slots = expanded_inputs[n_single:n_single + n_important]
    other_slots = expanded_inputs[n_single + n_important:]

    assignments = []

    def assign(slots, pool):
        if not pool:
            return
        for p in slots:
            yield p, random.choice(pool)

    if single_combs:
        assignments += list(assign(single_slots, single_combs))

    if important_combs:
        assignments += list(assign(important_slots, important_combs))

    if other_combs:
        assignments += list(assign(other_slots, other_combs))

    random.shuffle(assignments)

    if not assignments:
        raise RuntimeError(
            "任务分配为空。\n"
            "请检查 input_files、sampling_times、single_ratio、important_ratio 以及 ENABLED_COMBINATIONS。"
        )

    # ---------- 加载多源场景配置 ----------
    full_scene_configs_map = {}

    needed_dirs = set(MODULE_CONFIG_MAP.values())
    print("\n📂 正在执行物理隔离加载...")

    for d in sorted(needed_dirs):
        full_scene_configs_map[d] = safe_load_configs_from_path(d)
        print(
            f"  - 目录 '{d}': 成功从磁盘直接读取了 "
            f"{len(full_scene_configs_map[d])} 个场景文件"
        )

    # ---------- 参数隔离校验 ----------
    if "configs_comb" in full_scene_configs_map and "configs_comb_q" in full_scene_configs_map:
        val1 = (
            full_scene_configs_map["configs_comb"]
            .get("distortion", {})
            .get("params", {})
            .get("target_lufs")
        )
        val2 = (
            full_scene_configs_map["configs_comb_q"]
            .get("distortion", {})
            .get("params", {})
            .get("target_lufs")
        )

        print(f"\n🧪 参数隔离校验:")
        print(f"  - configs_comb   中的 distortion target_lufs: {val1}")
        print(f"  - configs_comb_q 中的 distortion target_lufs: {val2}")

        if val1 == val2:
            print("  ⚠️ 警告: 两个目录下的参数依然相同，请检查原始磁盘文件内容是否真的不一致！")
        else:
            print("  ✅ 隔离成功：不同目录下的同名场景已加载不同参数。")

    print("")

    # ---------- 构造任务 & 溯源打印 ----------
    all_tasks = []
    printed_combs = set()

    print("\n🚧 构造任务中...")

    for input_path, comb in assignments:
        base = os.path.splitext(os.path.basename(input_path))[0]
        meta = meta_dict.get(base)

        c_name = comb["name"]
        config_source = comb["config_source"]

        if config_source not in full_scene_configs_map:
            raise MissingConfigError(
                f"组合 {c_name} 指向的配置目录没有被加载。\n"
                f"config_source: {config_source}\n"
                f"已加载目录: {sorted(full_scene_configs_map.keys())}"
            )

        current_scene_configs = full_scene_configs_map[config_source]

        # DEBUG 追踪：每个组合仅打印一次完整信息
        if args.debug and c_name not in printed_combs:
            print(f"\n{'=' * 30}")
            print(f"🔍 [DEBUG SOURCE] 组合名: {c_name}")
            print(f"  - 来源模块: {comb['defined_in_module']}.py")
            print(f"  - 配置目录: {config_source}/")
            print(f"  - 包含场景: {comb['scenes']}")

            for sn in comb["scenes"]:
                content = current_scene_configs.get(sn, "未找到配置")
                detail = json.dumps(content, indent=4, ensure_ascii=False)
                print(f"    --- 场景 '{sn}' 的完整配置内容 ---")
                print(detail)

            print(f"\n{'=' * 30}\n")
            printed_combs.add(c_name)

        # 这里是关键改动：读不到 config 或 effect 直接 raise
        merged_chain = build_effect_chain_or_raise(current_scene_configs, comb)

        for v in range(args.num_variants):
            out_dir = os.path.join(execu.OUTPUT_DIR, c_name, base)
            fname = f"{base}_{c_name}_{args.mapper}_{v + 1}.wav"

            all_tasks.append({
                "input_path": input_path,
                "output_path": os.path.join(out_dir, fname),
                "effect_chain": merged_chain,
                "mapper_name": args.mapper,
                "meta_info": meta,
                "comb_name": c_name,
                "scene_names": comb["scenes"],
                "variant_id": v + 1,
                "base_name": base,
                "speed_augment": args.speed_augment,
            })

    print(f"\n✅ 任务构造完成，总计: {len(all_tasks)}")

    if args.debug:
        print("💡 [Tips] 请检查上方打印的配置片段，确认其 params 里的数值是否为修改后的值。")

    if not all_tasks:
        raise RuntimeError(
            "all_tasks 为空，无法继续执行。\n"
            "这通常说明 assignments 为空，或所有组合都没有成功构造 effect_chain。"
        )

    # ---------- 执行部分 ----------
    jsonl_entries = []

    with Pool(
        processes=min(args.workers, cpu_count()),
        initializer=execu.init_worker,
    ) as pool:
        for result in pool.imap_unordered(execu.process_single_task, all_tasks):
            if result.get("success") and result.get("jsonl_entry"):
                jsonl_entries.append(result["jsonl_entry"])

    if jsonl_entries:
        jsonl_entries.sort(key=lambda x: x["index"])

        with open(args.output_jsonl, "w", encoding="utf-8") as f:
            for entry in jsonl_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"✅ 写入增强 jsonl: {args.output_jsonl}")
    else:
        raise RuntimeError(
            "处理完成，但没有生成任何 jsonl entry。\n"
            "请检查 process_single_task 是否全部失败，建议打开 executor 内部日志。"
        )


if __name__ == "__main__":
    main()