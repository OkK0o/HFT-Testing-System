# fin_data_processor:

负责对数据进行清洗，可以增加新方法

# minute_factors:

负责计算分钟级别因子

# tick_factors:

负责计算tick级别因子

# factors_test:

计算IC及RankIC

# factors_analysis:

负责测试新因子

# Workflow:

1. 数据清洗
2. 分钟级别因子计算
3. tick级别因子计算
4. 因子测试
5. 因子分析
6. 合适因子纳入到分钟或tick级别因子计算中

# TODO:

 线性/非线性模型训练，回测分析，分布忘写了

# 有效性、分散性

# 刻画有效性的时候，20ticks，时间窗口取长一点。

1.因子分群
2.在不同枝加因子

算一下分位数的平均盈亏

# 计算Return用中间价
