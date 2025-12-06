import os

def get_script_name(file_path):
    """获取当前脚本的名称"""
    script_path = file_path
    script_name = os.path.basename(script_path)

    return script_name