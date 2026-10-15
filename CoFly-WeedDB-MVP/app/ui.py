"""
Gradio-based web UI for weed detection system.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional
import json
import tempfile
import numpy as np

from config import PROJECT_ROOT, ui_config, inference_config
from utils import setup_logging, load_image, save_image
from inference import WeedDetector
from tile_inference import TileBasedInference
from heatmap import HeatmapGenerator
from exporters import DetectionExporter

logger = setup_logging("UI", PROJECT_ROOT / "outputs" / "debug" / "ui.log")

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    logging.warning("gradio not installed. UI will not be available.")


class WeedDetectionUI:
    """Gradio UI for weed detection."""
    
    def __init__(self, model_path: str = "models/unified/best.pt"):
        """
        Initialize UI.
        
        Args:
            model_path: path to trained model
        """
        if not GRADIO_AVAILABLE:
            raise RuntimeError("gradio is not installed. Install via: pip install gradio")
        
        self.model_path = model_path
        self.detector = None
        self.tile_inf = None
        self.heatmap_gen = HeatmapGenerator()
        self.exporter = DetectionExporter()
        
        self._load_model()
    
    def _load_model(self):
        """Load detection model."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.detector = WeedDetector(self.model_path, conf_threshold=0.25)
            self.tile_inf = TileBasedInference(self.detector)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def detect_single_image(self, image: np.ndarray, conf_threshold: float) -> Tuple:
        """
        Run detection on single image.
        
        Args:
            image: input image
            conf_threshold: confidence threshold
            
        Returns:
            Tuple of (result_image, json_results, csv_results)
        """
        try:
            # Update threshold
            self.detector.conf_threshold = conf_threshold
            
            # Save temp image for detection
            temp_path = PROJECT_ROOT / "outputs" / ".temp_ui_image.png"
            save_image(image, str(temp_path))
            
            logger.info(f"Running detection on temp image: {temp_path}")
            
            # Run detection
            result = self.detector.detect_image(str(temp_path), save_visualization=True)
            
            if result is None:
                logger.error("Detection returned None")
                return image, "Detection failed", ""
            
            # Get visualization
            vis_img = result.get('visualization')
            if vis_img is None:
                logger.warning("Visualization is None, using original image")
                vis_img = image
            
            # Format results
            detections = result.get('detections', [])
            json_str = json.dumps(detections, indent=2)
            
            # Create CSV string
            if len(detections) > 0:
                csv_lines = ["class_id,class_name,confidence,x1,y1,x2,y2"]
                for det in detections:
                    x1, y1, x2, y2 = det['bbox_xyxy']
                    csv_lines.append(f"{det['class_id']},{det['class_name']},"
                                   f"{det['confidence']:.4f},{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}")
                csv_str = "\n".join(csv_lines)
            else:
                csv_str = "No detections found"
            
            logger.info(f"Detection complete: {len(detections)} detections found")
            
            # Clean up
            if temp_path.exists():
                temp_path.unlink()
            
            return vis_img, json_str, csv_str
        
        except Exception as e:
            logger.error(f"Detection error: {e}", exc_info=True)
            return image, f"Error: {str(e)}", ""
    
    def detect_large_image(self, image: np.ndarray, conf_threshold: float,
                          tile_size: int = 640, overlap: float = 0.3) -> Tuple:
        """
        Run tile-based detection on large image.
        
        Args:
            image: input image
            conf_threshold: confidence threshold
            tile_size: tile size
            overlap: tile overlap ratio
            
        Returns:
            Tuple of (bbox_image, heatmap_image, json_results, csv_results, status, state)
        """
        try:
            # Update threshold and tile config
            self.detector.conf_threshold = conf_threshold
            self.tile_inf = TileBasedInference(self.detector, tile_size=tile_size,
                                              overlap_ratio=overlap)
            
            # Save temp image
            temp_path = PROJECT_ROOT / "outputs" / ".temp_ui_large_image.png"
            save_image(image, str(temp_path))
            
            # Run tile inference
            result = self.tile_inf.infer_large_image(str(temp_path))
            
            if result is None:
                return None, None, "No detections", "", "", None
            
            # Get visualization
            vis_img = result['visualization']
            heatmap = self.heatmap_gen.generate_image_space_heatmap(
                result['detections'],
                image_size=result['image_size'],
                sigma=30.0,
                sensitivity=1.3,
            )
            heatmap_img = self.heatmap_gen.visualize_heatmap(
                heatmap,
                image=image,
                overlay_alpha=0.45,
            )
            
            # Format results
            detections = result['detections']
            json_str = json.dumps(detections, indent=2)
            
            # Create CSV string
            csv_lines = ["class_id,class_name,confidence,x1,y1,x2,y2,lat,lon"]
            for det in detections:
                x1, y1, x2, y2 = det['bbox_xyxy']
                lat = det.get('latitude', '')
                lon = det.get('longitude', '')
                csv_lines.append(f"{det['class_id']},{det['class_name']},"
                               f"{det['confidence']:.4f},{x1:.2f},{y1:.2f},"
                               f"{x2:.2f},{y2:.2f},{lat},{lon}")
            csv_str = "\n".join(csv_lines)
            
            # Clean up
            if temp_path.exists():
                temp_path.unlink()
            
            state = {
                "detections": detections,
                "image": image,
                "image_size": result['image_size'],
            }
            
            return (
                vis_img,
                heatmap_img,
                json_str,
                csv_str,
                f"Processed {result['tiles_processed']} tiles, detections: {len(detections)}",
                state,
            )
        
        except Exception as e:
            logger.error(f"Tile inference error: {e}")
            return None, None, f"Error: {str(e)}", "", "", None

    def render_large_image_heatmap(
        self,
        result_state: Optional[dict],
        heatmap_opacity: float = 0.4,
        heatmap_sigma: float = 30.0,
        heatmap_sensitivity: float = 1.0,
    ) -> Tuple:
        """Render heatmap from cached detections without rerunning inference."""
        if not result_state:
            return None, "Run large image processing first"
        
        detections = result_state.get("detections", [])
        image = result_state.get("image")
        image_size = tuple(result_state.get("image_size", (image.shape[1], image.shape[0])))
        
        heatmap = self.heatmap_gen.generate_image_space_heatmap(
            detections,
            image_size=image_size,
            sigma=heatmap_sigma,
            sensitivity=heatmap_sensitivity,
        )
        heatmap_img = self.heatmap_gen.visualize_heatmap(
            heatmap,
            image=image,
            overlay_alpha=heatmap_opacity,
        )
        return heatmap_img, f"Heatmap updated, detections: {len(detections)}"
    
    def create_interface(self):
        """Create Gradio interface."""
        import numpy as np
        
        with gr.Blocks(title="WeedDB Detections") as demo:
            gr.Markdown("# 🌾 Weed Detection System")
            gr.Markdown("UAV-based weed detection using YOLOv8")
            
            with gr.Tabs():
                # Tab 1: Single Image
                with gr.TabItem("Single Image"):
                    with gr.Row():
                        with gr.Column():
                            img_input = gr.Image(label="Upload Image", type="numpy")
                            conf_slider = gr.Slider(minimum=0.01, maximum=1.0, value=0.12,
                                                    label="Confidence Threshold")
                            detect_btn = gr.Button("Detect Weeds", variant="primary")
                        
                        with gr.Column():
                            vis_output = gr.Image(label="Detection Results")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### JSON Results")
                            json_output = gr.Textbox(label="Detections JSON", lines=10,
                                                    interactive=False)
                        
                        with gr.Column():
                            gr.Markdown("### CSV Results")
                            csv_output = gr.Textbox(label="Detections CSV", lines=10,
                                                   interactive=False)
                    
                    detect_btn.click(
                        self.detect_single_image,
                        inputs=[img_input, conf_slider],
                        outputs=[vis_output, json_output, csv_output]
                    )
                
                # Tab 2: Large Image (Tile-based)
                with gr.TabItem("Large Image"):
                    large_result_state = gr.State(None)
                    with gr.Row():
                        with gr.Column():
                            large_img_input = gr.Image(label="Upload Large Image", type="numpy")
                            gr.Markdown("### Detection Settings")
                            conf_slider2 = gr.Slider(minimum=0.01, maximum=1.0, value=0.12,
                                                    label="Confidence Threshold")
                            tile_size = gr.Slider(minimum=256, maximum=1024, value=640, step=64,
                                                 label="Tile Size")
                            overlap = gr.Slider(minimum=0.1, maximum=0.9, value=0.3, step=0.1,
                                              label="Tile Overlap Ratio")
                            detect_large_btn = gr.Button("Process Large Image", variant="primary")
                            gr.Markdown("### Heatmap Settings")
                            heatmap_opacity = gr.Slider(minimum=0.0, maximum=1.0, value=0.45,
                                                        step=0.05, label="Heatmap Opacity")
                            heatmap_sigma = gr.Slider(minimum=10, maximum=120, value=30, step=2,
                                                      label="Heatmap Blur / Radius")
                            heatmap_sensitivity = gr.Slider(minimum=0.5, maximum=3.0, value=1.3, step=0.1,
                                                            label="Heatmap Sensitivity")
                            update_heatmap_btn = gr.Button("Update Heatmap")
                        
                        with gr.Column():
                            large_vis_output = gr.Image(label="Bounding Boxes")
                            large_heatmap_output = gr.Image(label="Heatmap")
                            status_output = gr.Textbox(label="Status", interactive=False)
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### JSON Results")
                            large_json_output = gr.Textbox(label="Detections JSON", lines=10,
                                                          interactive=False)
                        
                        with gr.Column():
                            gr.Markdown("### CSV Results")
                            large_csv_output = gr.Textbox(label="Detections CSV", lines=10,
                                                         interactive=False)
                    
                    detect_large_btn.click(
                        self.detect_large_image,
                        inputs=[
                            large_img_input,
                            conf_slider2,
                            tile_size,
                            overlap,
                        ],
                        outputs=[
                            large_vis_output,
                            large_heatmap_output,
                            large_json_output,
                            large_csv_output,
                            status_output,
                            large_result_state,
                        ]
                    )

                    update_heatmap_btn.click(
                        self.render_large_image_heatmap,
                        inputs=[
                            large_result_state,
                            heatmap_opacity,
                            heatmap_sigma,
                            heatmap_sensitivity,
                        ],
                        outputs=[
                            large_heatmap_output,
                            status_output,
                        ]
                    )
                
                # Tab 3: About
                with gr.TabItem("About"):
                    gr.Markdown("""
                    ## 🌾 Weed Detection System MVP
                    
                    ### Features
                    - **Single Image Detection**: Detect weeds in individual UAV images
                    - **Large Image Processing**: Process large field orthomosaics using tile-based inference
                    - **Geographic Support**: Extract and visualize coordinates from image metadata
                    - **Multiple Export Formats**: JSON, CSV, HTML, YOLO txt
                    
                    ### Model Info
                    - Model: YOLOv8 Nano
                    - Training Data: CoFly-WeedDB
                    - Classes: Weed (unified mode)
                    
                    ### Citation
                    If you use this system, please cite the CoFly-WeedDB dataset.
                    """)
        
        return demo


def launch_ui(model_path: str = "models/unified/best.pt", share: bool = False, server_port: int = 7860):
    """Launch Gradio UI."""
    try:
        ui = WeedDetectionUI(model_path)
        demo = ui.create_interface()
        
        logger.info(f"Launching UI at 127.0.0.1:{server_port}")
        demo.launch(
            server_name="127.0.0.1",
            server_port=server_port,
            share=share,
            show_error=True
        )
    
    except Exception as e:
        logger.error(f"Failed to launch UI: {e}")
        raise


def main():
    """Main UI script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch weed detection UI")
    parser.add_argument("--model", default="models/unified/best.pt",
                       help="Path to model weights")
    parser.add_argument("--port", type=int, default=7860,
                       help="Port to run on")
    parser.add_argument("--share", action="store_true",
                       help="Share link (requires gradio pro)")
    
    args = parser.parse_args()
    
    # Launch UI with specified port
    launch_ui(model_path=args.model, share=args.share, server_port=args.port)


if __name__ == "__main__":
    import numpy as np
    main()
