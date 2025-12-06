import os
import onnx
from onnx import external_data_helper


def merging_onnx_structures_parameters(model_path: str, output_path: str, verify: bool=True):
    """
    合并 ONNX 结构体和参数文件
    Args:
        model_path: ONNX 模型路径
        output_path: 输出合并后的 ONNX 模型路径
        verify: 是否验证合并后的模型
    """
    # 1. 加载模型（会自动关联 .data 文件）
    model = onnx.load(model_path, load_external_data=True) # type: ignore
    # 2. 如果新版函数不存在，则手动内嵌数据
    if hasattr(external_data_helper, "convert_model_to_single_file"):
        # ✅ 新版 ONNX (>=1.14)
        external_data_helper.convert_model_to_single_file(model) # type: ignore
    else:
        # 🔁 旧版 ONNX 手动写入外部数据
        print("⚠️ 当前 onnx 版本不支持 convert_model_to_single_file，改用手动嵌入方式")
        external_data_helper.load_external_data_for_model(model, os.path.dirname(model_path)) # type: ignore
        # 清除 external_data 字段，使数据写入模型本体
        for tensor in model.graph.initializer:
            tensor.external_data.clear()
            tensor.data_location = onnx.TensorProto.DEFAULT # type: ignore
    # 3. 保存新文件
    onnx.save(model, output_path) # type: ignore
    print(f"✅ 合并完成：{output_path}")
    # 4. (可选) 验证模型可用性
    if verify:
        onnx.checker.check_model(onnx.load(output_path)) # type: ignore
        print("✅ 模型验证通过")