#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据库初始化脚本
用于初始化SQLite数据库表结构
"""
import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from src.infra.db.data_objects import Base, SessionFactory


def init_database():
    """初始化数据库表结构"""
    print("=" * 60)
    print("开始初始化数据库...")
    print("=" * 60)
    
    try:
        # 创建数据目录
        data_dir = os.path.join(project_root, "data")
        os.makedirs(data_dir, exist_ok=True)
        print(f"\n✓ 数据目录已创建: {data_dir}")
        
        # 创建SessionFactory实例
        factory = SessionFactory()
        engine = factory.get_engine()
        
        print(f"✓ 数据库连接信息: {engine.url}")
        
        # 创建所有表
        Base.metadata.create_all(engine)
        print("\n✓ 数据库表结构创建完成！")
        
        # 显示创建的表
        print("\n已创建的表：")
        for table_name in Base.metadata.tables.keys():
            print(f"  - {table_name}")
        
        print("\n" + "=" * 60)
        print("✅ 数据库初始化完成！")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 数据库初始化失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = init_database()
    sys.exit(0 if success else 1)
