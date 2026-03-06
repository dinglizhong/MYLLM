#!/usr/bin/env python
# 调试脚本：检查GRPOConfig的参数

try:
    from trl import GRPOConfig
    import inspect
    
    print("GRPOConfig参数信息：")
    sig = inspect.signature(GRPOConfig.__init__)
    print("参数列表：")
    for param_name, param in sig.parameters.items():
        if param_name != 'self':
            default_val = param.default if param.default != inspect.Parameter.empty else "NO_DEFAULT"
            print(f"  {param_name}: default = {default_val}")
            
    print("\nGRPOConfig文档：")
    print(GRPOConfig.__doc__[:500] if GRPOConfig.__doc__ else "无文档")
    
except Exception as e:
    print(f"错误: {e}")