#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera Ray Transformation Visualizer
====================================

A professional Streamlit application for visualizing camera intrinsic transformations
during crop and padding operations. This tool demonstrates ray invariance across
different camera configurations and provides interactive 3D visualizations.

Features:
---------
• Interactive camera intrinsic parameter adjustment
• Real-time crop and padding visualization
• 2D image overlays with principal point and crop regions
• 3D ray visualizations showing camera geometry
• Ray invariance validation across transformations
• Professional UI with clear parameter controls

Usage:
------
1. Upload an image or use the default checkerboard pattern
2. Adjust camera intrinsics (focal lengths, principal point)
3. Set crop parameters (position and size)
4. Configure padding values (left, right, top, bottom)
5. View real-time 2D and 3D visualizations
6. Monitor ray invariance validation metrics

Technical Details:
-----------------
• Camera Model: Pinhole camera with intrinsic matrix K
• Ray Computation: Inverse projection using K^(-1)
• Validation: Angular error between corresponding rays
• Visualization: Plotly for interactive 2D/3D plots

Author: Subhransu Bhattacharjee (1ssb)
Repository: https://github.com/1ssb/ray_visualizer.git
License: Apache-2.0
Date: January 2025
"""

import io
import numpy as np
from PIL import Image
import streamlit as st
import plotly.graph_objects as go
from typing import Tuple, Optional, List


class RayVisualizer:
    """
    A class for visualizing camera ray transformations and intrinsic matrices.
    
    This class provides utilities for:
    - Computing camera intrinsic matrices
    - Generating rays from pixel coordinates
    - Creating 2D and 3D visualizations
    - Validating ray invariance across transformations
    """
    
    def __init__(self):
        """Initialize the RayVisualizer."""
        self.z_vis = 10.0  # Visualization depth for 3D plots
    
    @staticmethod
    def clamp(value: float, low: float, high: float) -> float:
        """Clamp a value between low and high bounds."""
        return max(low, min(high, value))
    
    @staticmethod
    def create_checkerboard(height: int, width: int) -> np.ndarray:
        """
        Create a checkerboard pattern for testing.
        
        Args:
            height: Image height in pixels
            width: Image width in pixels
            
        Returns:
            Checkerboard image as uint8 array
        """
        y, x = np.ogrid[:height, :width]
        tile = (((x // 40) + (y // 40)) % 2) * 0.1 + 0.85
        img = np.stack([tile, tile, tile], axis=-1)
        return (img * 255).astype(np.uint8)
    
    @staticmethod
    def rays_from_pixels(K: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute rays from pixel coordinates using camera intrinsic matrix.
        
        Args:
            K: 3x3 camera intrinsic matrix
            u: Horizontal pixel coordinates
            v: Vertical pixel coordinates
            
        Returns:
            Normalized ray directions (N, 3)
        """
        uv1 = np.stack([u, v, np.ones_like(u)], axis=-1)  # (N, 3)
        Kinv = np.linalg.inv(K)
        d = (Kinv @ uv1.T).T
        n = np.linalg.norm(d, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return d / n
    
    @staticmethod
    def scale_to_z(d: np.ndarray, z: float) -> np.ndarray:
        """
        Scale ray directions to intersect a plane at depth z.
        
        Args:
            d: Ray directions (N, 3)
            z: Target depth
            
        Returns:
            3D points at depth z (N, 3)
        """
        s = np.where(np.abs(d[:, 2]) > 1e-12, z / d[:, 2], 0.0)
        return d * s[:, None]
    
    @staticmethod
    def make_plane_polygon(K: np.ndarray, width: int, height: int, z: float) -> np.ndarray:
        """
        Create a polygon representing the image plane at depth z.
        
        Args:
            K: Camera intrinsic matrix
            width: Image width
            height: Image height
            z: Depth of the plane
            
        Returns:
            Corner points of the plane (4, 3)
        """
        corners = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=float)
        d = RayVisualizer.rays_from_pixels(K, corners[:, 0], corners[:, 1])
        return RayVisualizer.scale_to_z(d, z)
    
    @staticmethod
    def create_intrinsic_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
        """
        Create camera intrinsic matrix from focal lengths and principal point.
        
        Args:
            fx: Focal length in x direction
            fy: Focal length in y direction
            cx: Principal point x coordinate
            cy: Principal point y coordinate
            
        Returns:
            3x3 camera intrinsic matrix
        """
        return np.array([[fx, 0.0, cx],
                        [0.0, fy, cy],
                        [0.0, 0.0, 1.0]], dtype=float)
    
    def compute_principal_shifted_matrices(
        self, 
        width: int, 
        height: int, 
        fx: float, 
        fy: float, 
        cx: float, 
        cy: float,
        crop_x: int, 
        crop_y: int, 
        crop_w: int, 
        crop_h: int,
        pad_left: int, 
        pad_right: int, 
        pad_top: int, 
        pad_bottom: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, int, int, int]:
        """
        Compute intrinsic matrices for original, cropped, and expanded images.
        
        Args:
            width, height: Original image dimensions
            fx, fy: Focal lengths
            cx, cy: Principal point coordinates
            crop_x, crop_y, crop_w, crop_h: Crop parameters
            pad_left, pad_right, pad_top, pad_bottom: Padding parameters
            
        Returns:
            Tuple of (K_orig, K_crop, K_exp, W_exp, H_exp, crop_x, crop_y, crop_w, crop_h)
        """
        # Clamp crop parameters
        crop_x = self.clamp(crop_x, 0, width - 1)
        crop_y = self.clamp(crop_y, 0, height - 1)
        crop_w = self.clamp(crop_w, 1, width - crop_x)
        crop_h = self.clamp(crop_h, 1, height - crop_y)
        
        # Original intrinsic matrix
        K_orig = self.create_intrinsic_matrix(fx, fy, cx, cy)
        
        # Cropped intrinsic matrix
        cx_crop = cx - crop_x
        cy_crop = cy - crop_y
        K_crop = self.create_intrinsic_matrix(fx, fy, cx_crop, cy_crop)
        
        # Expanded intrinsic matrix
        cx_exp = cx_crop + pad_left
        cy_exp = cy_crop + pad_top
        K_exp = self.create_intrinsic_matrix(fx, fy, cx_exp, cy_exp)
        
        # Expanded dimensions
        W_exp = crop_w + pad_left + pad_right
        H_exp = crop_h + pad_top + pad_bottom
        
        return (K_orig, K_crop, K_exp, W_exp, H_exp,
                crop_x, crop_y, crop_w, crop_h)
    
    @staticmethod
    def angular_error_degrees(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute angular error between two sets of unit vectors.
        
        Args:
            a, b: Unit vectors (N, 3)
            
        Returns:
            Angular errors in degrees (N,)
        """
        dot = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
        return np.degrees(np.arccos(dot))


class PlotlyVisualizer:
    """Handles creation of Plotly figures for 2D and 3D visualizations."""
    
    @staticmethod
    def create_image_with_overlays(
        img: np.ndarray, 
        width: int, 
        height: int, 
        K: np.ndarray, 
        title: str, 
        show_crop: bool = False, 
        crop: Optional[Tuple[int, int, int, int]] = None, 
        extra_rect: Optional[Tuple[int, int, int, int]] = None
    ) -> go.Figure:
        """
        Create a 2D image visualization with overlays.
        
        Args:
            img: Image array (H, W, 3)
            width, height: Image dimensions
            K: Camera intrinsic matrix
            title: Figure title
            show_crop: Whether to show crop rectangle
            crop: Crop parameters (x, y, w, h)
            extra_rect: Extra rectangle parameters (x, y, w, h)
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        fig.add_trace(go.Image(z=img))
        fig.update_yaxes(autorange='reversed')
        fig.update_xaxes(range=[0, width], constrain='domain')
        fig.update_yaxes(range=[0, height], scaleanchor="x", scaleratio=1)
        
        shapes = []
        
        # Frame
        shapes.append(dict(
            type="rect", x0=0, y0=0, x1=width, y1=height,
            line=dict(color="#333", width=1.2)
        ))
        
        # Principal axes and point
        cx, cy = float(K[0, 2]), float(K[1, 2])
        shapes.append(dict(
            type="line", x0=cx, y0=0, x1=cx, y1=height,
            line=dict(color="#d62728", width=1, dash="dash")
        ))
        shapes.append(dict(
            type="line", x0=0, y0=cy, x1=width, y1=cy,
            line=dict(color="#d62728", width=1, dash="dash")
        ))
        
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy], mode="markers",
            marker=dict(size=10, color="#d62728", symbol="star"),
            showlegend=False
        ))
        
        # Crop rectangle
        if show_crop and crop is not None:
            x0, y0, w, h = crop
            shapes.append(dict(
                type="rect", x0=x0, y0=y0, x1=x0+w, y1=y0+h,
                line=dict(color="#10a37f", width=2, dash="dash")
            ))
        
        # Extra rectangle
        if extra_rect is not None:
            x0, y0, w, h = extra_rect
            shapes.append(dict(
                type="rect", x0=x0, y0=y0, x1=x0+w, y1=y0+h,
                line=dict(color="#10a37f", width=2, dash="dash")
            ))
        
        fig.update_layout(
            title=title,
            margin=dict(l=10, r=10, t=35, b=10),
            xaxis=dict(showticklabels=False, zeroline=False),
            yaxis=dict(showticklabels=False, zeroline=False),
            shapes=shapes,
            height=360
        )
        return fig
    
    @staticmethod
    def create_3d_rays(
        K: np.ndarray, 
        width: int, 
        height: int, 
        highlight_uv: Optional[Tuple[np.ndarray, np.ndarray]], 
        title: str, 
        z_vis: float = 10.0,
        show_context: bool = True
    ) -> go.Figure:
        """
        Create a 3D visualization of camera rays.
        
        Args:
            K: Camera intrinsic matrix
            width, height: Image dimensions
            highlight_uv: Pixel coordinates to highlight (u, v)
            title: Figure title
            z_vis: Visualization depth
            show_context: Whether to show context rays or only highlight rays
            
        Returns:
            Plotly 3D figure object
        """
        def lines_xyz(P: np.ndarray) -> Tuple[List[float], List[float], List[float]]:
            """Convert points to line segments with NaN separators."""
            xs, ys, zs = [], [], []
            for i in range(P.shape[0]):
                xs += [0, P[i, 0], np.nan]
                ys += [0, P[i, 1], np.nan]
                zs += [0, P[i, 2], np.nan]
            return xs, ys, zs
        
        fig = go.Figure()
        
        # Context rays (only if show_context is True)
        if show_context:
            grid_n = 7
            uu = np.linspace(0, width - 1, grid_n)
            vv = np.linspace(0, height - 1, grid_n)
            UU, VV = np.meshgrid(uu, vv)
            d_ctx = RayVisualizer.rays_from_pixels(K, UU.ravel(), VV.ravel())
            P_ctx = RayVisualizer.scale_to_z(d_ctx, z_vis)
            x_c, y_c, z_c = lines_xyz(P_ctx)
            
            fig.add_trace(go.Scatter3d(
                x=x_c, y=y_c, z=z_c, mode='lines',
                line=dict(width=2, color='rgba(120,120,120,0.3)'),
                name="context rays", showlegend=False
            ))
        
        # Highlight rays (main focus)
        if highlight_uv is not None and len(highlight_uv[0]) > 0:
            u_h, v_h = highlight_uv
            d = RayVisualizer.rays_from_pixels(K, u_h, v_h)
            P = RayVisualizer.scale_to_z(d, z_vis)
            x_h, y_h, z_h = lines_xyz(P)
            fig.add_trace(go.Scatter3d(
                x=x_h, y=y_h, z=z_h, mode='lines',
                line=dict(width=6, color='#10a37f'),
                name="focus rays", showlegend=False
            ))
        
        # Camera center
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0], mode='markers',
            marker=dict(size=8, color='#d62728'), name="camera"
        ))
        
        # Principal ray
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[0, z_vis], mode='lines',
            line=dict(width=8, color='#d62728'), name="principal"
        ))
        
        # Image plane
        plane = RayVisualizer.make_plane_polygon(K, width, height, z_vis)
        fig.add_trace(go.Mesh3d(
            x=plane[:, 0], y=plane[:, 1], z=plane[:, 2],
            i=[0, 0], j=[1, 2], k=[2, 3],
            opacity=0.2, color='#1f77b4', name="image plane", showlegend=False
        ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                aspectmode='cube',
                xaxis=dict(showgrid=True, zeroline=False),
                yaxis=dict(showgrid=True, zeroline=False),
                zaxis=dict(showgrid=True, zeroline=False, range=[0, z_vis*1.05])
            ),
            margin=dict(l=10, r=10, t=35, b=10),
            height=520
        )
        return fig


class StreamlitApp:
    """Main Streamlit application class."""
    
    def __init__(self):
        """Initialize the Streamlit application."""
        self.ray_visualizer = RayVisualizer()
        self.plotly_visualizer = PlotlyVisualizer()
        self.setup_page()
    
    def setup_page(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Crop+Pad Ray Visualizer", 
            layout="wide",
            page_icon="📷"
        )
        st.title("📷 Camera Ray Transformation Visualizer")
        st.markdown("---")
    
    def load_image(self) -> np.ndarray:
        """Load image from file upload or create default checkerboard."""
        with st.sidebar:
            st.title("Controls")
            
            # Image input
            uploaded_file = st.file_uploader(
                "Load image (RGB)", 
                type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
            )
            
            if uploaded_file:
                img = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
                return np.array(img)
            else:
                return self.ray_visualizer.create_checkerboard(480, 640)
    
    def get_camera_parameters(self, width: int, height: int) -> Tuple[float, float, float, float]:
        """Get camera intrinsic parameters from sidebar."""
        with st.sidebar:
            st.subheader("📐 Camera Intrinsics")
            
            enforce_square = st.checkbox("Enforce fx = fy", True)
            default_f = 0.8 * max(width, height)
            
            fx = st.number_input(
                "fx (focal length x)", 
                value=float(default_f), 
                min_value=1.0, 
                step=10.0, 
                format="%.3f"
            )
            fy = st.number_input(
                "fy (focal length y)", 
                value=float(default_f), 
                min_value=1.0, 
                step=10.0, 
                format="%.3f"
            )
            
            if enforce_square:
                fy = fx
            
            cx = st.number_input(
                "cx (principal point x)", 
                value=float(width/2.0), 
                min_value=0.0, 
                max_value=float(width), 
                step=1.0, 
                format="%.3f"
            )
            cy = st.number_input(
                "cy (principal point y)", 
                value=float(height/2.0), 
                min_value=0.0, 
                max_value=float(height), 
                step=1.0, 
                format="%.3f"
            )
            
            return fx, fy, cx, cy
    
    def get_crop_parameters(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Get crop parameters from sidebar."""
        with st.sidebar:
            st.subheader("✂️ Crop Parameters")
            
            crop_x = st.slider("Crop X", 0, width-1, int(width*0.25))
            crop_y = st.slider("Crop Y", 0, height-1, int(height*0.25))
            crop_w = st.slider("Crop Width", 1, width, int(width*0.5))
            crop_h = st.slider("Crop Height", 1, height, int(height*0.5))
            
            # Center crop action
            if st.button("🎯 Center 50% crop"):
                crop_w = int(width*0.5)
                crop_h = int(height*0.5)
                crop_x = (width - crop_w) // 2
                crop_y = (height - crop_h) // 2
            
            return crop_x, crop_y, crop_w, crop_h
    
    def get_padding_parameters(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Get padding parameters from sidebar."""
        with st.sidebar:
            st.subheader("📏 Padding Parameters")
            
            max_pad = max(width, height)
            pad_left = st.slider("Pad Left", 0, max_pad, int(width*0.16))
            pad_right = st.slider("Pad Right", 0, max_pad, int(width*0.16))
            pad_top = st.slider("Pad Top", 0, max_pad, int(height*0.16))
            pad_bottom = st.slider("Pad Bottom", 0, max_pad, int(height*0.16))
            
            return pad_left, pad_right, pad_top, pad_bottom
    
    def validate_ray_invariance(self, K_orig: np.ndarray, K_crop: np.ndarray, K_exp: np.ndarray,
                               u_orig: np.ndarray, v_orig: np.ndarray,
                               u_crop: np.ndarray, v_crop: np.ndarray,
                               u_exp: np.ndarray, v_exp: np.ndarray) -> Tuple[bool, float, float]:
        """Validate that rays remain invariant across transformations."""
        d_orig = self.ray_visualizer.rays_from_pixels(K_orig, u_orig, v_orig)
        d_crop = self.ray_visualizer.rays_from_pixels(K_crop, u_crop, v_crop)
        d_exp = self.ray_visualizer.rays_from_pixels(K_exp, u_exp, v_exp)
        
        max_err_crop = float(np.max(self.ray_visualizer.angular_error_degrees(d_orig, d_crop)))
        max_err_exp = float(np.max(self.ray_visualizer.angular_error_degrees(d_orig, d_exp)))
        
        ok = (max_err_crop < 1e-10) and (max_err_exp < 1e-10)
        return ok, max_err_crop, max_err_exp
    
    def display_status(self, ok: bool, max_err_crop: float, max_err_exp: float,
                      K_orig: np.ndarray, K_crop: np.ndarray, K_exp: np.ndarray):
        """Display validation status and intrinsic matrices."""
        col_a, col_b = st.columns([2, 3])
        
        with col_a:
            if ok:
                st.success(
                    f"✅ Ray invariance verified\n"
                    f"Max Δθ orig↔crop: {max_err_crop:.3e}°\n"
                    f"Max Δθ orig↔expanded: {max_err_exp:.3e}°"
                )
            else:
                st.error(
                    f"❌ Rays changed\n"
                    f"Max Δθ orig↔crop: {max_err_crop:.3e}°\n"
                    f"Max Δθ orig↔expanded: {max_err_exp:.3e}°"
                )
        
        with col_b:
            st.code(
                f"K_original=\n{K_orig}\n\n"
                f"K_cropped=\n{K_crop}\n\n"
                f"K_expanded=\n{K_exp}",
                language="text"
            )
    
    def create_derived_images(self, img: np.ndarray, crop_x: int, crop_y: int, 
                             crop_w: int, crop_h: int, W_exp: int, H_exp: int,
                             pad_left: int, pad_top: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create cropped and expanded images."""
        crop_img = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w].copy()
        exp_img = np.full((H_exp, W_exp, 3), [245, 248, 252], dtype=np.uint8)
        exp_img[pad_top:pad_top+crop_h, pad_left:pad_left+crop_w] = crop_img
        
        return crop_img, exp_img
    
    def run(self):
        """Run the main application."""
        # Load image
        img = self.load_image()
        height, width = img.shape[:2]
        
        # Get parameters
        fx, fy, cx, cy = self.get_camera_parameters(width, height)
        crop_x, crop_y, crop_w, crop_h = self.get_crop_parameters(width, height)
        pad_left, pad_right, pad_top, pad_bottom = self.get_padding_parameters(width, height)
        
        # Compute transformations
        (K_orig, K_crop, K_exp, W_exp, H_exp,
         crop_x, crop_y, crop_w, crop_h) = self.ray_visualizer.compute_principal_shifted_matrices(
            width, height, fx, fy, cx, cy,
            crop_x, crop_y, crop_w, crop_h,
            pad_left, pad_right, pad_top, pad_bottom
        )
        
        # Create derived images
        crop_img, exp_img = self.create_derived_images(
            img, crop_x, crop_y, crop_w, crop_h, W_exp, H_exp, pad_left, pad_top
        )
        
        # Sample points for validation (crop region)
        sample_n = 5
        u_s = np.linspace(crop_x + 0.5, crop_x + crop_w - 0.5, sample_n)
        v_s = np.linspace(crop_y + 0.5, crop_y + crop_h - 0.5, sample_n)
        UU, VV = np.meshgrid(u_s, v_s)
        u_orig, v_orig = UU.ravel(), VV.ravel()
        u_crop = u_orig - crop_x
        v_crop = v_orig - crop_y
        u_exp = u_crop + pad_left
        v_exp = v_crop + pad_top
        
        # Generate rays for the full expanded plane
        grid_n_exp = 8
        u_exp_full = np.linspace(0, W_exp - 1, grid_n_exp)
        v_exp_full = np.linspace(0, H_exp - 1, grid_n_exp)
        UU_exp, VV_exp = np.meshgrid(u_exp_full, v_exp_full)
        u_exp_full_rays = UU_exp.ravel()
        v_exp_full_rays = VV_exp.ravel()
        
        # Validate ray invariance
        ok, max_err_crop, max_err_exp = self.validate_ray_invariance(
            K_orig, K_crop, K_exp, u_orig, v_orig, u_crop, v_crop, u_exp, v_exp
        )
        
        # Display status
        self.display_status(ok, max_err_crop, max_err_exp, K_orig, K_crop, K_exp)
        
        # 2D visualizations
        st.subheader("📊 2D Image Views")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig1 = self.plotly_visualizer.create_image_with_overlays(
                img, width, height, K_orig,
                title=f"Original (f=({fx:.0f},{fy:.0f}), cx={K_orig[0,2]:.1f}, cy={K_orig[1,2]:.1f})",
                show_crop=True, crop=(crop_x, crop_y, crop_w, crop_h)
            )
            st.plotly_chart(fig1, use_container_width=True, config=dict(displayModeBar=False))
        
        with col2:
            fig2 = self.plotly_visualizer.create_image_with_overlays(
                crop_img, crop_w, crop_h, K_crop,
                title=f"Cropped (f=({fx:.0f},{fy:.0f}), cx={K_crop[0,2]:.1f}, cy={K_crop[1,2]:.1f})",
                show_crop=False
            )
            st.plotly_chart(fig2, use_container_width=True, config=dict(displayModeBar=False))
        
        with col3:
            fig3 = self.plotly_visualizer.create_image_with_overlays(
                exp_img, W_exp, H_exp, K_exp,
                title=f"Expanded (f=({fx:.0f},{fy:.0f}), cx={K_exp[0,2]:.1f}, cy={K_exp[1,2]:.1f})",
                show_crop=False, extra_rect=(pad_left, pad_top, crop_w, crop_h)
            )
            st.plotly_chart(fig3, use_container_width=True, config=dict(displayModeBar=False))
        
        # 3D visualizations
        st.subheader("🌐 3D Ray Visualizations")
        col4, col5, col6 = st.columns(3)
        
        with col4:
            st.plotly_chart(
                self.plotly_visualizer.create_3d_rays(
                    K_orig, width, height, (u_orig, v_orig), 
                    "3D Rays — Original (All Rays)", self.ray_visualizer.z_vis, show_context=True
                ),
                use_container_width=True
            )
        
        with col5:
            st.plotly_chart(
                self.plotly_visualizer.create_3d_rays(
                    K_crop, crop_w, crop_h, (u_crop, v_crop), 
                    "3D Rays — Cropped (Crop Region Only)", self.ray_visualizer.z_vis, show_context=False
                ),
                use_container_width=True
            )
        
        with col6:
            st.plotly_chart(
                self.plotly_visualizer.create_3d_rays(
                    K_exp, W_exp, H_exp, (u_exp_full_rays, v_exp_full_rays), 
                    "3D Rays — Expanded (Full Expanded Plane)", self.ray_visualizer.z_vis, show_context=False
                ),
                use_container_width=True
            )


def main():
    """Main entry point for the application."""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
