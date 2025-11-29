# 🚀 Application Run করার সহজ Guide

## ✅ Quick Start (সবচেয়ে সহজ)

### Method 1: Command Prompt/Terminal থেকে

1. **Command Prompt বা PowerShell খুলুন**

2. **Project folder এ যান:**
   ```bash
   cd c:\parameter_pruning_project
   ```

3. **এই command run করুন:**
   ```bash
   python -m streamlit run streamlit_app.py
   ```

4. **Browser automatically open হবে** `http://localhost:8501` এ

---

## Method 2: Batch File দিয়ে (Windows)

1. **`run_app.bat` file টা double-click করুন**
   - Automatically সব setup হবে
   - Dashboard browser এ open হবে

---

## Method 3: Python Script দিয়ে

1. **Terminal এ:**
   ```bash
   python run_app.py
   ```

---

## 📋 Step-by-Step (যদি প্রথমবার run করছেন)

### Step 1: Dependencies Check করুন

Terminal এ এই command run করুন:
```bash
python -m pip list
```

যদি `streamlit`, `torch`, `matplotlib` না থাকে, তাহলে install করুন:
```bash
python -m pip install torch torchvision matplotlib numpy scikit-learn streamlit tqdm reportlab
python -m pip install "pillow<13,>=7.1.0" "altair<6,>=4.0,!=5.4.0,!=5.4.1"
python -m pip install streamlit --no-deps
python -m pip install altair blinker cachetools click pandas protobuf pydeck requests tenacity toml tornado watchdog gitpython jsonschema narwhals
```

### Step 2: Application Run করুন

```bash
python -m streamlit run streamlit_app.py
```

### Step 3: Browser এ Dashboard দেখুন

- Automatically browser open হবে
- URL: `http://localhost:8501`
- যদি না হয়, manually browser এ `http://localhost:8501` type করুন

---

## 🎯 Dashboard ব্যবহার

1. **🏠 Home**: Overview দেখুন
2. **🎯 Train Model**: 
   - Epochs, batch size, learning rate set করুন
   - "Start Training" click করুন
   - Model `saved/` folder এ save হবে
3. **📊 Visualize Model**: Model weights visualize করুন
4. **✂️ Prune Model**: Model prune করুন
5. **📈 Compare Models**: 2টা model compare করুন
6. **📁 Model Manager**: সব saved models manage করুন

---

## ⚠️ Common Issues & Solutions

### Issue 1: "streamlit is not recognized"
**Solution:**
```bash
python -m streamlit run streamlit_app.py
```
(`streamlit` এর পরিবর্তে `python -m streamlit` use করুন)

### Issue 2: Port already in use
**Solution:**
```bash
python -m streamlit run streamlit_app.py --server.port 8502
```

### Issue 3: Module not found
**Solution:**
```bash
python -m pip install streamlit torch torchvision matplotlib numpy
```

### Issue 4: Browser automatically open হয় না
**Solution:**
- Manually browser এ যান: `http://localhost:8501`
- অথবা terminal output এ URL দেখুন

---

## 🛑 Application Stop করতে

Terminal এ **Ctrl+C** press করুন

---

## ✅ Checklist

- [ ] Python installed (check: `python --version`)
- [ ] Dependencies installed
- [ ] Command run করেছি: `python -m streamlit run streamlit_app.py`
- [ ] Browser এ dashboard দেখতে পাচ্ছি

---

## 📝 Quick Commands Summary

```bash
# 1. Project folder এ যান
cd c:\parameter_pruning_project

# 2. Run করুন
python -m streamlit run streamlit_app.py

# 3. Browser এ যান
# http://localhost:8501
```

---

**Happy Coding! 🎉**

