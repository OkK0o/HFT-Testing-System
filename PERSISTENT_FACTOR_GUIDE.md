# 持久化因子管理系统使用指南

## 概述

持久化因子管理系统是对现有因子管理系统的扩展，旨在解决重复注册因子的问题。通过将因子元数据保存到CSV文件中，可以避免每次程序运行时都需要重新注册所有因子。该系统提供以下优势：

1. **避免重复注册** - 不需要每次都调用 `register_all_factors()`
2. **代码精简** - 可以将因子定义与使用分离
3. **更好的可维护性** - 因子信息集中管理，易于查看和修改
4. **向后兼容** - 与现有系统无缝集成，最小化代码修改

## 系统组件

该系统由以下几个主要组件组成：

1. **`persistent_factor_manager.py`** - 持久化因子管理器核心实现
2. **`factor_manager_adapter.py`** - 适配器，提供统一接口
3. **`import_factors_to_persistent.py`** - 工具脚本，将现有因子导入到持久化系统
4. **`persistent_factor_example.py`** - 示例脚本，展示完整工作流程

## 快速开始

### 1. 准备工作

将系统文件复制到项目目录：

```bash
# 核心文件
cp persistent_factor_manager.py /path/to/your/project/
cp factor_manager_adapter.py /path/to/your/project/
cp import_factors_to_persistent.py /path/to/your/project/
```

### 2. 导入现有因子到持久化系统

```bash
# 导入因子
python import_factors_to_persistent.py

# 指定输出CSV文件
python import_factors_to_persistent.py --output my_factors.csv

# 指定注册函数
python import_factors_to_persistent.py --register-module factor_definitions --register-func register_all_factors
```

### 3. 修改现有代码

将原有代码中的 `FactorManager` 导入修改为使用适配器：

```python
# 原始代码
from factor_manager import FactorManager, FactorFrequency

# 修改为
from factor_manager_adapter import FactorManager, FactorFrequency
```

初始化适配器（在程序开始处）：

```python
from factor_manager_adapter import FactorManagerProxy

# 默认情况下会自动检测是否使用持久化管理器
# 如果factor_metadata.csv存在且非空，则使用持久化管理器
FactorManagerProxy.initialize()

# 强制使用持久化管理器
FactorManagerProxy.initialize(use_persistent=True)

# 指定元数据CSV文件
FactorManagerProxy.initialize(metadata_csv='my_factors.csv')
```

### 4. 运行示例

```bash
# 注册示例因子并使用持久化管理器
python persistent_factor_example.py --register-factors --use-persistent --calculate-factors --demo-data

# 导入因子到持久化系统并计算
python persistent_factor_example.py --import-factors --use-persistent --calculate-factors --input data.feather --output result.feather
```

## 详细使用说明

### 持久化因子管理器 (PersistentFactorManager)

`PersistentFactorManager` 提供与原始 `FactorManager` 相同的接口，但会将因子元数据保存到CSV文件中。主要方法包括：

- `register()` - 注册因子
- `register_smoothed_factor()` - 注册平滑因子
- `get_factor_names()` - 获取因子名称
- `get_factor_info()` - 获取因子信息
- `calculate_factors()` - 计算因子
- `get_factor_frequency()` - 获取因子频率

示例：

```python
from persistent_factor_manager import PersistentFactorManager, FactorFrequency

# 注册因子
@PersistentFactorManager.registry.register(
    name="momentum_5",
    frequency=FactorFrequency.MINUTE,
    category="momentum",
    description="5周期动量因子"
)
def calculate_momentum_5(df):
    result = df.copy()
    result['momentum_5'] = result.groupby('InstruID')['LastPrice'].pct_change(5)
    return result

# 注册平滑因子
PersistentFactorManager.register_smoothed_factor("momentum_5", 3, "ema")

# 获取因子信息
factor_info = PersistentFactorManager.get_factor_info()
print(factor_info)

# 计算因子
result = PersistentFactorManager.calculate_factors(
    df, FactorFrequency.MINUTE, ["momentum_5", "momentum_5_ema3"]
)
```

### 因子管理器适配器 (FactorManagerProxy)

`FactorManagerProxy` 提供统一的接口，可以根据配置自动选择使用原始 `FactorManager` 还是 `PersistentFactorManager`。使用方式与原始 `FactorManager` 完全一致。

控制使用哪个管理器的方式：

1. **环境变量**：设置 `USE_PERSISTENT_FACTOR_MANAGER=true`
2. **初始化参数**：`FactorManagerProxy.initialize(use_persistent=True)`
3. **自动检测**：如果未指定，会检查CSV文件是否存在

```python
from factor_manager_adapter import FactorManager, FactorFrequency, FactorManagerProxy

# 初始化适配器
FactorManagerProxy.initialize(use_persistent=True, metadata_csv='my_factors.csv')

# 使用方式与原FactorManager一致
@FactorManager.register(
    name="example_factor",
    frequency=FactorFrequency.MINUTE,
    category="example",
    description="示例因子"
)
def calculate_example_factor(df):
    df['example_factor'] = 1.0
    return df

# 计算因子
result = FactorManager.calculate_factors(df, FactorFrequency.MINUTE, ['example_factor'])
```

### 配置选项

可以通过以下方式配置持久化因子管理系统：

1. **环境变量**：

   ```bash
   # 使用持久化管理器
   export USE_PERSISTENT_FACTOR_MANAGER=true

   # 指定元数据CSV文件
   export FACTOR_METADATA_CSV=my_factors.csv
   ```
2. **初始化参数**：

   ```python
   # 在代码中指定
   FactorManagerProxy.initialize(
       use_persistent=True,
       metadata_csv='my_factors.csv'
   )
   ```

## 最佳实践

1. **定期备份元数据文件** - CSV文件包含所有因子信息，建议定期备份
2. **分组管理因子** - 使用不同的CSV文件管理不同类别的因子，例如：

   ```python
   # 动量因子
   FactorManagerProxy.initialize(metadata_csv='momentum_factors.csv')

   # 波动率因子
   FactorManagerProxy.initialize(metadata_csv='volatility_factors.csv')
   ```
3. **注册函数分离** - 将因子注册函数与使用代码分离，只在需要添加新因子时调用
4. **使用适配器而非直接使用** - 优先使用 `factor_manager_adapter.py` 中的接口，而不是直接使用 `persistent_factor_manager.py`
5. **逐步迁移** - 可以逐步将现有代码迁移到持久化系统，不需要一次性修改所有代码

## 常见问题

**Q: 我修改了因子实现，但计算结果没有变化**

A: 持久化系统只保存因子元数据，不保存函数实现。如果修改了函数实现，需要重新注册因子或者确保修改后的函数能够被正确导入。

**Q: 如何删除已注册的因子？**

A: 直接编辑元数据CSV文件，删除对应的行，或者使用以下代码：

```python
import pandas as pd

# 读取CSV
factors_df = pd.read_csv('factor_metadata.csv')

# 删除特定因子
factors_df = factors_df[factors_df['name'] != 'factor_to_delete']

# 保存回CSV
factors_df.to_csv('factor_metadata.csv', index=False)
```

**Q: 持久化系统支持所有现有功能吗？**

A: 是的，持久化系统与原始系统提供相同的接口，支持所有现有功能，包括因子注册、平滑因子、依赖关系管理等。

**Q: CSV文件格式是什么？**

A: CSV文件包含以下列：

- `name`: 因子名称
- `category`: 因子类别
- `description`: 因子描述
- `dependencies`: 依赖的其他因子（逗号分隔）
- `frequency`: 因子频率
- `registered_time`: 首次注册时间
- `last_updated`: 最后更新时间
- `smoothing_info`: 平滑因子信息（JSON格式）

## 深入了解

### 元数据文件结构

CSV文件包含因子的元数据，但不包含函数实现。当需要计算因子时，系统会:

1. 从CSV文件加载因子元数据
2. 检查函数是否已注册（通过原始注册或动态导入）
3. 如果函数可用，调用函数计算因子
4. 如果是平滑因子，动态计算（不需要函数定义）

### 系统兼容性

该系统设计为向后兼容，如果发现以下情况，会自动回退到使用原始 `FactorManager`：

1. 持久化管理器不可用（`persistent_factor_manager.py` 缺失）
2. 元数据CSV文件不存在或为空
3. 明确指定不使用持久化管理器

### 自定义扩展

如需扩展系统功能，可以参考以下几个方向：

1. **支持更多存储格式** - 除了CSV，可以支持JSON、SQLite等
2. **添加版本控制** - 跟踪因子变更历史
3. **增加更多元数据** - 如性能指标、使用统计等
4. **实现自动测试** - 自动测试因子有效性
