# Camera Ray Transformation Visualizer

A professional Streamlit application for visualizing camera intrinsic transformations during crop and padding operations. This tool demonstrates ray invariance across different camera configurations and provides interactive 3D visualizations.

![Ray Visualizer](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)

## 🌟 Features

- **Interactive Camera Controls**: Adjust focal lengths, principal point, and other intrinsic parameters
- **Real-time Visualization**: See changes instantly as you modify parameters
- **2D Image Overlays**: Visualize principal point, crop regions, and image boundaries
- **3D Ray Visualizations**: Interactive 3D plots showing camera geometry and ray directions
- **Ray Invariance Validation**: Monitor how rays remain consistent across transformations
- **Professional UI**: Clean, intuitive interface with organized parameter controls

## 🚀 Quick Start

### Local Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/1ssb/ray_visualizer.git
   cd ray_visualizer
   ```
2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application**:

   ```bash
   streamlit run app.py
   ```
4. **Open your browser** and navigate to `http://localhost:8501`

## 📖 Usage Guide

### 1. Image Input

- Upload your own image (PNG, JPG, JPEG, BMP, TIF, TIFF)
- Or use the default checkerboard pattern for testing

### 2. Camera Intrinsics

- **fx, fy**: Focal lengths in pixels
- **cx, cy**: Principal point coordinates
- **Enforce fx = fy**: Option to maintain square pixels

### 3. Crop Parameters

- **Crop X, Y**: Position of the crop region
- **Crop Width, Height**: Size of the crop region
- **Center 50% crop**: Quick button for centered cropping

### 4. Padding Parameters

- **Pad Left/Right/Top/Bottom**: Padding values in pixels
- Adjust to see how the expanded image affects camera geometry

### 5. Visualizations

- **2D Views**: Original, cropped, and expanded images with overlays
- **3D Views**: Interactive ray visualizations for each transformation
- **Validation**: Real-time ray invariance metrics

## 🔧 Technical Details

### Camera Model

The application uses a pinhole camera model with intrinsic matrix K:

```
K = [fx  0  cx]
    [0   fy cy]
    [0   0   1]
```

### Ray Computation

Rays are computed using inverse projection:

```
d = K^(-1) * [u, v, 1]^T
```

where (u,v) are pixel coordinates.

### Ray Invariance

The tool validates that rays remain invariant across transformations by computing angular errors between corresponding rays in different coordinate systems.

## 🏗️ Architecture

The application is built with a modular, class-based architecture:

- **`RayVisualizer`**: Core ray computation and camera geometry
- **`PlotlyVisualizer`**: 2D and 3D visualization components
- **`StreamlitApp`**: Main application logic and UI

## 🚀 Deployment

### Streamlit Cloud

1. Fork this repository
2. Connect your GitHub account to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy the app using the repository URL

### Heroku

1. Add a `Procfile`:
   ```
   web: streamlit run ray.py --server.port=$PORT --server.address=0.0.0.0
   ```
2. Deploy using Heroku CLI or GitHub integration

### Local Production

```bash
streamlit run ray.py --server.port=8501 --server.address=0.0.0.0
```

## 📁 Project Structure

```
ray_visualizer/
├── ray.py              # Main application file
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── Dockerfile         # Docker configuration
└── .gitignore         # Git ignore rules
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Subhransu Bhattacharjee (1ssb)**

- GitHub: [@1ssb](https://github.com/1ssb)
- Repository: [ray_visualizer](https://github.com/1ssb/ray_visualizer)

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the web interface
- 3D visualizations powered by [Plotly](https://plotly.com/)
- Image processing with [Pillow](https://python-pillow.org/)
- Numerical computations with [NumPy](https://numpy.org/)

---

⭐ **Star this repository if you find it useful!**
