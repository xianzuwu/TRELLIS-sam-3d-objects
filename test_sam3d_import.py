#!/usr/bin/env python
"""
测试脚本：验证 sam3d_objects 模块可以正确导入和使用
"""
import sys
import os

# 确保当前目录在 Python 路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print("=" * 60)
print("测试 sam3d_objects 模块导入")
print("=" * 60)

# 测试基本导入
try:
    import sam3d_objects
    print("✓ sam3d_objects 模块导入成功")
except ImportError as e:
    print(f"✗ sam3d_objects 导入失败: {e}")
    sys.exit(1)

# 测试各个子模块
modules_to_test = [
    'sam3d_objects.model',
    'sam3d_objects.pipeline',
    'sam3d_objects.utils',
    'sam3d_objects.config',
    'sam3d_objects.data',
]

print("\n测试子模块导入:")
for module_name in modules_to_test:
    try:
        __import__(module_name)
        print(f"  ✓ {module_name}")
    except ImportError as e:
        print(f"  ✗ {module_name}: {e}")

# 测试具体类的导入（根据实际需要调整）
print("\n测试具体类导入:")
try:
    from sam3d_objects.model import *
    print("  ✓ sam3d_objects.model.*")
except ImportError as e:
    print(f"  ✗ sam3d_objects.model.*: {e}")

try:
    from sam3d_objects.pipeline import *
    print("  ✓ sam3d_objects.pipeline.*")
except ImportError as e:
    print(f"  ✗ sam3d_objects.pipeline.*: {e}")

print("\n" + "=" * 60)
print("所有测试完成！")
print("=" * 60)
print("\n使用示例:")
print("  from sam3d_objects.model import ...")
print("  from sam3d_objects.pipeline import ...")
print("  from sam3d_objects.utils import ...")

