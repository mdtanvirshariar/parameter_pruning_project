# 🚀 Application Run করার জন্য Step-by-Step Guide

## Windows এ Run করার জন্য:

### Method 1: সবচেয়ে সহজ (Recommended) ⭐

1. **`run_app.bat` file টা double-click করুন**
   - Automatically সব dependencies install হবে
   - Dashboard automatically browser এ open হবে

### Method 2: Manual Step-by-Step

#### Step 1: Python Check করুন
```bash
python --version
```
Python 3.7+ থাকতে হবে। না থাকলে Python install করুন।

#### Step 2: Virtual Environment তৈরি করুন (Optional কিন্তু Recommended)
```bash
python -m venv venv
venv\Scripts\activate
```

#### Step 3: Dependencies Install করুন
```bash
pip install -r requirements.txt
```
এটা install করবে:
- torch, torchvision
- streamlit
- matplotlib, numpy
- tqdm
- scikit-learn

#### Step 4: Application Run করুন
```bash
streamlit run streamlit_app.py
```
অথবা
```bash
python run_app.py
```

#### Step 5: Browser এ Dashboard দেখুন
- Automatically browser এ open হবে
- URL: `http://localhost:8501`

---

## Linux/Mac এ Run করার জন্য:

### Method 1: Script ব্যবহার করুন
```bash
chmod +x run_app.sh
./run_app.sh
```

### Method 2: Manual
```bash
# Virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run streamlit_app.py
```

---

## 🎯 Dashboard ব্যবহার করার জন্য:

1. **🏠 Home**: Overview দেখুন
2. **🎯 Train Model**: নতুন model train করুন
   - Epochs, batch size, learning rate set করুন
   - "Start Training" button click করুন
3. **📊 Visualize Model**: Model weights visualize করুন
4. **✂️ Prune Model**: Model prune করুন
5. **📈 Compare Models**: 2টা model compare করুন
6. **📁 Model Manager**: সব saved models manage করুন

---

## ⚠️ Troubleshooting:

### Problem: "Module not found" error
**Solution**: 
```bash
pip install -r requirements.txt
```

### Problem: Port already in use
**Solution**: 
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Problem: Windows এ num_workers error
**Solution**: Already fixed! Code automatically Windows detect করে num_workers=0 use করবে।

### Problem: Model load করতে পারছেন না
**Solution**: 
- প্রথমে একটা model train করুন
- Model `saved/` folder এ save হবে

---

## 📝 Quick Start Commands:

```bash
# 1. Dependencies install
pip install -r requirements.txt

# 2. Run dashboard
streamlit run streamlit_app.py

# OR use the launcher (Windows)
run_app.bat

# OR use Python launcher
python run_app.py
```

---

## ✅ Checklist:

- [ ] Python installed (3.7+)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Run command executed
- [ ] Browser automatically opened
- [ ] Dashboard visible at http://localhost:8501

---

**Happy Coding! 🎉**

