import gradio as gr
from llm import run_agent_steps


def process_image_and_text(uploaded_file, question):
    if not question.strip():
        raise gr.Error("请输入你的问题。")
    if not uploaded_file:
        raise gr.Error("请先上传一个多波段 GeoTIFF 文件。")

    print("[INFO] 收到 1 个多波段文件，开始预处理与计算")
    reply, images = "", []
    try:
        for reply, images in run_agent_steps(question, [uploaded_file]):
            # 生成阶段不动输入框，避免用户正在输入时被清空
            yield reply, images, gr.skip(), gr.skip()
    except Exception as e:
        yield f"处理失败：{e}", [], gr.skip(), gr.skip()
        return

    # 结束时：出图则连文件一起清；反问/失败则保留文件，便于接着回答
    yield reply, images, "", (None if images else gr.skip())


with gr.Blocks(title="RS-Agent L2 (多指数版)") as demo:
    gr.Markdown("# 🛰️ 遥感智能体 L2 - 多指数热力图")
    gr.Markdown("⚠️ 数据准备提示：请在上传前，使用 ENVI、QGIS 或 Python 将单波段文件融合成多波段文件。系统将自动读取并进行预处理。")
    gr.Markdown(
        "支持 **NDVI / NDWI / NDBI / RVI / SAVI / NBR / BSI / NDTI / NBR2** 等标准指数，"
        "以及 **LAI（经验式估算）** 和 **chl_proxy（自制代理）** 两个非标准指标。"
        "系统按你的问题自动路由：直接点名或说特征即触发对应指数；多场景冲突时最多计算最核心的 2 个，"
        "其余以“建议”提示，不会一次跑满全部指数。"
    )
    gr.Markdown(
        "> **使用说明**：本工具只反映该期影像的光谱特征，未做定量反演与精度验证，"
        "结果仅供初步参考，不能作为决策、执法或评估依据；判断变化/退化需要至少两期影像。"
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.File(
                label="上传多波段影像（.tif / .tiff）",
                file_types=[".tif", ".tiff"],
                file_count="single"
            )
            question_input = gr.Textbox(
                label="输入指令",
                placeholder="例如：这里生态环境怎么样？假设这是 Landsat 8 影像。",
                lines=3
            )
            submit_btn = gr.Button("🚀 发送请求", variant="primary")
        with gr.Column(scale=2):
            output_text = gr.Textbox(
                label="模型分析结论",
                lines=10,
                max_lines=24,
                interactive=False,
                placeholder="统计摘要、掩膜提示、缺失波段说明和顾问结论会完整显示在这里。"
            )
            output_gallery = gr.Gallery(
                label="指数热力图",
                columns=2,
                height=700,
                object_fit="contain",
                preview=True
            )

    submit_btn.click(
        process_image_and_text,
        [image_input, question_input],
        [output_text, output_gallery, question_input, image_input]
    )

if __name__ == "__main__":
    demo.launch()
