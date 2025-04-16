#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导入现有因子到持久化系统
======================

将当前注册的因子导入到持久化CSV系统中，避免每次运行都需要重新注册
"""

import argparse
import sys
from pathlib import Path
import importlib
import inspect
import pandas as pd
from typing import List, Dict

def find_factor_register_modules(search_dir: str = '.', pattern: str = 'factor_*.py') -> List[str]:
    """查找可能包含因子注册的模块文件"""
    from pathlib import Path
    
    search_path = Path(search_dir)
    module_files = list(search_path.glob(pattern))
    
    module_names = []
    for module_file in module_files:
        if module_file.name.startswith('__'):
            continue
        module_names.append(module_file.stem)
    
    return module_names

def import_module_dynamically(module_name: str):
    """动态导入模块"""
    return importlib.import_module(module_name)

def find_register_functions(module) -> List[str]:
    """查找模块中的注册函数"""
    register_funcs = []
    
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and name.lower().find('register') != -1 and name.lower().find('factor') != -1:
            register_funcs.append(name)
    
    return register_funcs

def execute_register_function(module, func_name: str):
    """执行注册函数"""
    func = getattr(module, func_name)
    return func()

def sync_to_persistent():
    """将因子从FactorManager同步到PersistentFactorManager"""
    from factor_manager import FactorManager
    from persistent_factor_manager import PersistentFactorManager
    
    # 获取当前注册的因子信息
    factor_info = FactorManager.get_factor_info()
    
    imported_count = 0
    for _, row in factor_info.iterrows():
        name = row['name']
        
        # 检查因子是否已经在持久化系统中
        persistent_factors = PersistentFactorManager.get_factor_names()
        if name in persistent_factors:
            print(f"因子 '{name}' 已经存在于持久化系统中，跳过")
            continue
        
        # 平滑因子处理
        if 'smoothed' in row['category']:
            # 从平滑信息解析原始因子、窗口大小和方法
            smooth_info = row['smoothing_info']
            if not smooth_info:
                print(f"警告: 平滑因子 '{name}' 没有平滑信息，跳过")
                continue
            
            # 解析平滑信息
            import re
            match = re.search(r'平滑自: (.*?), 窗口: (\d+), 方法: (.*?)$', smooth_info)
            if match:
                original_factor = match.group(1)
                window = int(match.group(2))
                method = match.group(3)
                
                # 注册平滑因子
                try:
                    PersistentFactorManager.register_smoothed_factor(
                        original_factor, window, method
                    )
                    imported_count += 1
                    print(f"已导入平滑因子 '{name}'")
                except Exception as e:
                    print(f"导入平滑因子 '{name}' 时出错: {str(e)}")
            else:
                print(f"警告: 无法解析平滑因子 '{name}' 的信息，跳过")
        
        # 其他类型因子的处理需要原始函数，无法直接从FactorManager导入
        # 这些因子需要通过执行原始注册函数来导入
    
    print(f"共导入 {imported_count} 个因子到持久化系统")
    return imported_count

def main():
    parser = argparse.ArgumentParser(description='导入现有因子到持久化系统')
    parser.add_argument('--search-dir', type=str, default='.', help='搜索模块的目录')
    parser.add_argument('--pattern', type=str, default='factor_*.py', help='模块文件名匹配模式')
    parser.add_argument('--register-module', type=str, help='指定包含注册函数的模块名')
    parser.add_argument('--register-func', type=str, help='指定注册函数名')
    parser.add_argument('--output', type=str, default='factor_metadata.csv', help='输出的CSV文件路径')
    parser.add_argument('--force', action='store_true', help='强制重新导入所有因子')
    args = parser.parse_args()
    
    # 设置输出文件路径
    from persistent_factor_manager import PersistentFactorRegistry
    PersistentFactorRegistry.metadata_file = args.output
    
    # 导入和注册因子
    if args.register_module and args.register_func:
        # 使用指定的模块和函数
        try:
            module = import_module_dynamically(args.register_module)
            execute_register_function(module, args.register_func)
            print(f"已执行 {args.register_module}.{args.register_func}()")
        except Exception as e:
            print(f"执行指定函数时出错: {str(e)}")
            return
    else:
        # 搜索并执行所有注册函数
        modules = find_factor_register_modules(args.search_dir, args.pattern)
        print(f"找到可能包含因子注册的模块: {modules}")
        
        for module_name in modules:
            try:
                module = import_module_dynamically(module_name)
                register_funcs = find_register_functions(module)
                
                print(f"模块 {module_name} 中找到的注册函数: {register_funcs}")
                
                for func_name in register_funcs:
                    try:
                        execute_register_function(module, func_name)
                        print(f"已执行 {module_name}.{func_name}()")
                    except Exception as e:
                        print(f"执行 {module_name}.{func_name}() 时出错: {str(e)}")
            except Exception as e:
                print(f"导入模块 {module_name} 时出错: {str(e)}")
    
    # 将因子同步到持久化系统
    try:
        sync_to_persistent()
    except Exception as e:
        print(f"同步因子到持久化系统时出错: {str(e)}")
    
    # 创建持久化适配器
    print("\n创建持久化因子管理器适配器...")
    from persistent_factor_manager import PersistentFactorManager
    
    # 输出导入的因子信息
    factor_info = PersistentFactorManager.get_factor_info()
    print(f"\n持久化系统中的因子 ({len(factor_info)}):")
    print(factor_info[['name', 'category', 'frequency']])
    
    print(f"\n因子元数据已保存到 {args.output}")
    print("\n完成! 现在可以使用 PersistentFactorManager 代替 FactorManager")

if __name__ == "__main__":
    main() 