import functools
import gradio as gr
from gradio_imageslider import ImageSlider


def run_demo_server():
    # infer_gpu = spaces.GPU(functools.partial(infer))
    gradio_theme = gr.themes.Default()

    with gr.Blocks(
        theme=gradio_theme,
        title=f"DA^2",
        css="""
            #download {
                height: 118px;
            }
            .slider .inner {
                width: 5px;
                background: #FFF;
            }
            .viewport {
                aspect-ratio: 4/3;
            }
            .tabs button.selected {
                font-size: 20px !important;
                color: crimson !important;
            }
            h1 {
                text-align: center;
                display: block;
            }
            h2 {
                text-align: center;
                display: block;
            }
            h3 {
                text-align: center;
                display: block;
            }
            .md_feedback li {
                margin-bottom: 0px !important;
            }
        """,
        head="""
            <script async src="https://www.googletagmanager.com/gtag/js?id=G-1FWSVCGZTG"></script>
            <script>
                window.dataLayer = window.dataLayer || [];
                function gtag() {dataLayer.push(arguments);}
                gtag('js', new Date());
                gtag('config', 'G-1FWSVCGZTG');
            </script>
        """,
    ) as demo:
        gr.Markdown(
            """
            # DA<sup>2</sup>: <u>D</u>epth <u>A</u>nything in <u>A</u>ny <u>D</u>irection
            <p align="center">
            <a title="Project Page" href="https://depth-any-in-any-dir.github.io/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                <img src="https://img.shields.io/badge/Project-Website-pink?logo=googlechrome&logoColor=white">
            </a>
            <a title="arXiv" href="http://arxiv.org/abs/2509.26618" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white">
            </a>
            <a title="Github" href="https://github.com/EnVision-Research/DA-2" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                <img src="https://img.shields.io/github/stars/EnVision-Research/DA-2?label=GitHub%20%E2%98%85&logo=github&color=C8C" alt="badge-github-stars">
            </a>
            <a title="Social" href="https://x.com/_akhaliq/status/1973283687652606411" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                <img src="https://www.obukhov.ai/img/badges/badge-social.svg" alt="social">
            </a>
            <a title="Social" href="https://x.com/haodongli00/status/1973287870317338747" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                <img src="https://www.obukhov.ai/img/badges/badge-social.svg" alt="social">
            </a>
            <br>
            <strong>Please consider starring <span style="color: orange">&#9733;</span> our <a href="https://github.com/EnVision-Research/DA-2" target="_blank" rel="noopener noreferrer">GitHub Repo</a> if you find this demo useful!</strong>
        """
        )
        with gr.Tabs(elem_classes=["tabs"]):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(
                        label="Input Image",
                        type="filepath",
                    )
                    with gr.Row():
                        input_mask = gr.Image(
                            label="Input Mask (All pixels are assumed to be valid if mask is None)",
                            type="filepath",
                        )
                    with gr.Row():
                        image_submit_btn = gr.Button(
                            value=f"Get 360° depth!", variant="primary"
                        )
                        image_reset_btn = gr.Button(value="Reset")

                with gr.Column():
                    output_3dpc = ImageSlider(
                        label="Output 3D Points",
                        type="filepath",
                        interactive=False,
                        elem_classes="slider",
                        position=0.25,
                    )
                    with gr.Row():
                        output_depth = ImageSlider(
                            label="Output Depth",
                            type="filepath",
                            interactive=False,
                            elem_classes="slider",
                            position=0.25,
                        )
                    with gr.Row():
                        output_normal = ImageSlider(
                            label="Output Normal",
                            type="filepath",
                            interactive=False,
                            elem_classes="slider",
                            position=0.25,
                        )

        ### Server launch
        demo.queue(
            api_open=False,
        ).launch(
            server_name="0.0.0.0",
            server_port=7860,
        )

def main():
    run_demo_server()

if __name__ == "__main__":
    main()
