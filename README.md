
# Determining Parameter Redundancy and Pruning in Deep Neural Networks by Visualization Methods

**Project for BSc (CSE)**
Author: Shariar Rabbi (use your name as needed)
Dataset used: CIFAR-10 (default). You can change to MNIST by editing configs.

This repository contains code, visualization tools and a Streamlit dashboard to:
- Train a CNN on CIFAR-10
- Detect parameter redundancy (magnitude, activation sparsity, correlation)
- Apply magnitude-based and structured pruning
- Visualize before/after weight distributions, filter heatmaps, t-SNE of activations
- Generate report and presentation

## Structure
- `src/` : core scripts (model, training, pruning, visualization)
- `streamlit_app.py` : interactive dashboard to visualize models and pruning effects
- `requirements.txt` : Python dependencies
- `report.pdf` : short academic-style report (summary)
- `LICENSE` : MIT

## Quick start (local)

### Easy Way (Recommended) - Run Everything Together

**Windows:**
```bash
run_app.bat
```

**Linux/Mac:**
```bash
./run_app.sh
# Or
python run_app.py
```

This will automatically:
- Check and install dependencies
- Create necessary directories
- Launch the Streamlit dashboard

### Manual Way

1. Create and activate virtualenv (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Option A: Use the Dashboard (Recommended)**
   ```bash
   streamlit run streamlit_app.py
   # Or use the launcher:
   python run_app.py
   ```
   The dashboard now includes:
   - 🎯 **Train Model**: Train new models directly from the UI
   - 📊 **Visualize Model**: View weight distributions and statistics
   - ✂️ **Prune Model**: Apply pruning with real-time feedback
   - 📈 **Compare Models**: Compare before/after pruning effects
   - 📁 **Model Manager**: Manage all your saved models

3. **Option B: Use Command Line**
   ```bash
   # Train baseline model:
   python src/train.py --epochs 10 --save-dir saved
   
   # Apply pruning:
   python src/prune.py --model-path saved/baseline.pth --prune-percent 0.4 --save-dir saved
   
   # Launch dashboard:
   streamlit run streamlit_app.py
   ```

## New Features

### Enhanced Dashboard
- **Multi-page interface** with navigation sidebar
- **Model training** directly from the UI with progress tracking
- **Real-time statistics** showing parameters, sparsity, file size
- **Model comparison** with side-by-side metrics and accuracy charts
- **Model manager** to view, evaluate, and delete saved models
- **Better visualizations** with improved layout and display

### Unified Startup
- **One-click launch** with `run_app.bat` (Windows) or `run_app.sh` (Linux/Mac)
- Automatic dependency checking and installation
- Directory creation and setup
- Cross-platform support

## Notes
- Code defaults to CIFAR-10 and PyTorch. If you prefer TensorFlow, adapt model and training functions.
- The included scripts are well-commented to help you customize for your university submission.
- The dashboard automatically handles model loading, evaluation, and visualization.
- All models are saved in the `saved/` directory and visualizations in `assets/`.
